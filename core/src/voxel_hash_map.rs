use itertools::Itertools;
use nalgebra::{
    Dyn, Isometry3, Matrix, Matrix3x6, Matrix6, RawStorage, RawStorageMut, Storage, Vector3,
    Vector6, U1, U3,
};
use nalgebra_lapack::Cholesky;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    ops::{
        sophus::{Exp, Hat},
        transform::Transform,
    },
    types::{IntoIsometry3, VoxelPoint},
};

#[derive(Clone, Debug)]
pub struct VoxelHashMap {
    args: VoxelHashMapArgs,
    storage: VoxelHashMapStorage,
    voxel_size: f64,
}

impl VoxelHashMap {
    #[inline]
    pub fn new(args: VoxelHashMapArgs, voxel_size: f64) -> Self {
        Self {
            args,
            storage: VoxelHashMapStorage::default(),
            voxel_size,
        }
    }

    pub fn from_global_points(points: &[VoxelPoint], voxel_size: f64) -> Self {
        let max_points_per_voxel = usize::MAX;

        Self {
            args: VoxelHashMapArgs {
                max_distance2: f64::MAX,
                max_points_per_voxel,
            },
            storage: {
                let mut storage = VoxelHashMapStorage::default();
                points
                    .iter()
                    .filter_map(|point| {
                        (point / voxel_size)
                            .try_cast::<i32>()
                            .map(|voxel| (Voxel(voxel), *point))
                    })
                    .for_each(|(voxel, point)| {
                        storage
                            .entry(voxel)
                            .and_modify(|block| block.add_point_unchecked(point))
                            .or_insert_with(|| VoxelBlock::new(point, max_points_per_voxel));
                    });
                storage
            },
            voxel_size,
        }
    }

    pub fn add_points<S>(&mut self, points: &Matrix<f64, Dyn, U3, S>)
    where
        S: RawStorage<f64, Dyn, U3>,
    {
        points
            .row_iter()
            .map(|point| point.transpose())
            .filter_map(|point| {
                (point / self.voxel_size)
                    .try_cast::<i32>()
                    .map(|voxel| (Voxel(voxel), point))
            })
            .for_each(|(voxel, point)| {
                self.storage
                    .entry(voxel)
                    .and_modify(|block| block.add_point(point))
                    .or_insert_with(|| VoxelBlock::new(point, self.args.max_points_per_voxel));
            })
    }

    #[inline]
    pub fn clear(&mut self) {
        self.storage.clear()
    }

    pub fn get_correspondences<'a, S>(
        &'a self,
        points: &'a Matrix<f64, Dyn, U3, S>,
        max_correspondence_distance: f64,
    ) -> impl 'a + ParallelIterator<Item = (VoxelPoint, VoxelPoint)>
    where
        S: Send + Sync + RawStorage<f64, Dyn, U3>,
    {
        // Lambda Function to obtain the KNN of one point, maybe refactor
        let get_closest_neighbor = |point: VoxelPoint| {
            let voxel = (point / self.voxel_size).try_cast::<i32>()?;

            let range = |x| (x - 1)..=(x + 1);
            let voxels = (range(voxel.x))
                .cartesian_product(range(voxel.y))
                .cartesian_product(range(voxel.z))
                .map(|((x, y), z)| Voxel(Vector3::new(x, y, z)));

            let neighbors = voxels
                .filter_map(|voxel| self.storage.get(&voxel))
                .map(|block| &block.points)
                .filter(|points| !points.is_empty())
                .flatten();

            neighbors
                .min_by_key(|&neighbor| OrderedFloat((neighbor - point).norm_squared()))
                .copied()
        };

        (0..points.nrows())
            .into_par_iter()
            .map(|i| points.row(i).transpose())
            // 1st lambda: Parallel computation
            .filter_map(move |point| {
                get_closest_neighbor(point).map(|closest_neighbor| (point, closest_neighbor))
            })
            .filter(move |(point, closest_neighbor)| {
                (point - closest_neighbor).norm() < max_correspondence_distance
            })
    }

    #[inline]
    pub fn get_point_cloud(&self) -> impl '_ + Iterator<Item = &VoxelPoint> {
        self.storage.values().flat_map(|block| &block.points)
    }

    #[inline]
    pub fn into_point_cloud(self) -> impl Iterator<Item = VoxelPoint> {
        self.storage.into_values().flat_map(|block| block.points)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn register_frame<S>(
        &self,
        frame: Matrix<f64, Dyn, U3, S>,
        initial_guess: impl IntoIsometry3,
        max_correspondence_distance: f64,
        kernel: f64,
    ) -> Isometry3<f64>
    where
        S: Send + Sync + RawStorageMut<f64, Dyn, U3>,
    {
        const ESTIMATION_THRESHOLD: f64 = 1e-4;
        const MAX_NUM_ITERATIONS: usize = 500;

        let initial_guess = initial_guess.into_isometry3();
        if self.is_empty() {
            return initial_guess;
        }

        // Equation (9)
        let mut source = frame;
        source.transform_mut(initial_guess);

        // ICP-loop
        let j = 0..MAX_NUM_ITERATIONS;
        let mut t_icp = Isometry3::default();
        for _ in j {
            // Equation (10)
            let corres = self.get_correspondences(&source, max_correspondence_distance);

            // Equation (11)
            let (j_tj, j_tr) = build_linear_system(corres, kernel);
            let dx = match Cholesky::new(j_tj) {
                Some(decompositor) => decompositor.solve(&(-j_tr)).unwrap(),
                None => break,
            };
            let estimation = dx.exp();

            // Equation (12)
            source.transform_mut(estimation);
            t_icp.transform_mut(estimation);

            // Termination criteria
            if dx.norm() < ESTIMATION_THRESHOLD {
                break;
            }
        }

        // Spit the final transformation
        let mut final_guess = initial_guess;
        final_guess.transform_mut(t_icp);
        final_guess
    }

    pub fn remove_points_far_from_location<S>(&mut self, location: &Matrix<f64, U3, U1, S>)
    where
        S: Storage<f64, U3, U1>,
    {
        let pruned: Vec<_> = self
            .storage
            .iter()
            .filter_map(|(voxel, block)| block.points.first().map(|point| (voxel, *point)))
            .filter(|(_, sample_point)| {
                (sample_point - location).norm_squared() > self.args.max_distance2
            })
            .map(|(&voxel, _)| voxel)
            .collect();

        for voxel in pruned {
            self.storage.remove(&voxel);
        }
    }

    #[inline]
    pub fn update_with_origin<Sp, So>(
        &mut self,
        points: &Matrix<f64, Dyn, U3, Sp>,
        origin: &Matrix<f64, U3, U1, So>,
    ) where
        Sp: RawStorage<f64, Dyn, U3>,
        So: Storage<f64, U3, U1>,
    {
        self.add_points(points);
        self.remove_points_far_from_location(origin);
    }

    pub fn update_with_pose<S>(
        &mut self,
        points: &Matrix<f64, Dyn, U3, S>,
        pose: impl Copy + IntoIsometry3,
    ) where
        S: Storage<f64, Dyn, U3>,
    {
        let pose = pose.into_isometry3();

        let mut points_transformed = Matrix::<f64, Dyn, U3, _>::zeros(points.nrows());
        for i in 0..points_transformed.nrows() {
            points_transformed.row_mut(i).copy_from(&{
                let mut point_t = points.row(i).transpose().clone_owned();
                point_t.transform_mut(pose);
                point_t.transpose()
            });
        }
        let origin = pose.translation.vector;
        self.update_with_origin(&points_transformed, &origin)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VoxelHashMapArgs {
    pub max_distance2: f64,
    pub max_points_per_voxel: usize,
}

impl Default for VoxelHashMapArgs {
    #[inline]
    fn default() -> Self {
        Self {
            max_distance2: Self::default_max_distance2(),
            max_points_per_voxel: Self::default_max_points_per_voxel(),
        }
    }
}

impl VoxelHashMapArgs {
    #[inline]
    const fn default_max_distance2() -> f64 {
        f64::MAX
    }

    #[inline]
    const fn default_max_points_per_voxel() -> usize {
        usize::MAX
    }
}

type VoxelHashMapStorage = ::hashbrown::HashMap<Voxel, VoxelBlock>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
struct Voxel(pub(crate) Vector3<i32>);

#[derive(Clone, Debug)]
struct VoxelBlock {
    /// buffer of points with a max limit of max_points
    points: Vec<VoxelPoint>,
    max_points: usize,
}

impl VoxelBlock {
    #[inline]
    fn new(point: VoxelPoint, max_points: usize) -> Self {
        Self {
            points: vec![point],
            max_points,
        }
    }

    #[inline]
    fn add_point(&mut self, point: VoxelPoint) {
        if self.points.len() < self.max_points {
            self.add_point_unchecked(point)
        }
    }

    #[inline]
    fn add_point_unchecked(&mut self, point: VoxelPoint) {
        self.points.push(point)
    }
}

fn build_linear_system(
    corres: impl ParallelIterator<Item = (VoxelPoint, VoxelPoint)>,
    kernel: f64,
) -> (Matrix6<f64>, Vector6<f64>) {
    let compute_jacobian_and_residual = |(source, target): (VoxelPoint, VoxelPoint)| {
        let redisual = source - target;

        let mut j_r = Matrix3x6::identity();
        j_r.fixed_columns_mut::<3>(3)
            .copy_from(&(-1.0 * source.hat()));
        (j_r, redisual)
    };

    corres
        .map(|corr| {
            let square = |x| x * x;
            let weight = |residual2| square(kernel) / square(kernel + residual2);

            let (j_r, residual) = compute_jacobian_and_residual(corr);
            let w = weight(residual.norm_squared());

            let j_t = j_r.transpose();
            let j_tw = j_t * w;
            let j_tj = j_tw * j_r;
            let j_tr = j_tw * residual;
            (j_tj, j_tr)
        })
        // 2nd Lambda: Parallel reduction of the private Jacboians
        .reduce(Default::default, |(j_tj_a, j_tr_a), (j_tj_b, j_tr_b)| {
            (j_tj_a + j_tj_b, j_tr_a + j_tr_b)
        })
}
