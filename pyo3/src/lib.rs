#![allow(non_snake_case)]

use kiss_icp_core::{
    deskew, metrics, preprocessing,
    threshold::AdaptiveThreshold,
    types::{IntoIsometry3, IsometryArray3},
    voxel_hash_map::{VoxelHashMap, VoxelHashMapArgs},
};
use numpy::{
    nalgebra::{Dyn, MatrixXx3, U1, U3, U4},
    PyArray2, ToPyArray,
};
use pyo3::{
    pyclass, pyfunction, pymethods, pymodule, types::PyModule, wrap_pyfunction, PyObject, PyResult,
    Python,
};
use rayon::iter::ParallelIterator;

type PyVoxelPoint<'py> = ::numpy::PyReadonlyArray1<'py, f64>;
type PyListVoxelPoint<'py> = ::numpy::PyReadonlyArray2<'py, f64>;

type PyIsometryMatrix3<'py> = ::numpy::PyReadonlyArray2<'py, f64>;

#[pyfunction]
fn _Vector3dVector(vec: PyObject) -> PyObject {
    vec
}

/// Map representation
#[pyclass]
struct _VoxelHashMap(VoxelHashMap);

#[pymethods]
impl _VoxelHashMap {
    #[new]
    fn new(voxel_size: f64, max_distance: f64, max_points_per_voxel: usize) -> Self {
        let args = VoxelHashMapArgs {
            max_distance2: max_distance * max_distance,
            max_points_per_voxel,
        };

        Self(VoxelHashMap::new(args, voxel_size))
    }

    fn _clear(&mut self) {
        self.0.clear()
    }

    fn _empty(&self) -> bool {
        self.0.is_empty()
    }

    fn _update(&mut self, points: PyListVoxelPoint, pose: PyIsometryMatrix3) {
        self._update_with_pose(points, pose)
    }

    #[inline]
    fn _update_with_origin(&mut self, points: PyListVoxelPoint, origin: PyVoxelPoint) {
        let points = points.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
        let origin = origin.try_as_matrix::<U3, U1, Dyn, Dyn>().unwrap();
        self.0.update_with_origin(&points, &origin)
    }

    #[inline]
    fn _update_with_pose(&mut self, points: PyListVoxelPoint, pose: PyIsometryMatrix3) {
        let points = points.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
        let pose = pose.try_as_matrix::<U4, U4, Dyn, Dyn>().unwrap();
        self.0.update_with_pose(&points, pose)
    }

    fn _add_points(&mut self, points: PyListVoxelPoint) {
        let points = points.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
        self.0.add_points(&points)
    }

    fn _remove_far_away_points(&mut self, origin: PyVoxelPoint) {
        let origin = origin.try_as_matrix::<U3, U1, Dyn, Dyn>().unwrap();
        self.0.remove_points_far_from_location(&origin)
    }

    fn _point_cloud<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let points: Vec<_> = self
            .0
            .get_point_cloud()
            .map(|point| point.transpose())
            .collect();
        MatrixXx3::from_rows(&points).to_pyarray(py)
    }

    fn _get_correspondences<'py>(
        &self,
        py: Python<'py>,
        points: PyListVoxelPoint<'py>,
        max_correspondance_distance: f64,
    ) -> (&'py PyArray2<f64>, &'py PyArray2<f64>) {
        let points = points.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
        let (points, closest_neighbors): (Vec<_>, Vec<_>) = self
            .0
            .get_correspondences(&points, max_correspondance_distance)
            .map(|(point, closest_neighbor)| (point.transpose(), closest_neighbor.transpose()))
            .unzip();
        (
            MatrixXx3::from_rows(&points).to_pyarray(py),
            MatrixXx3::from_rows(&closest_neighbors).to_pyarray(py),
        )
    }
}

/// Point Cloud registration
#[pyfunction]
fn _register_point_cloud<'py>(
    py: Python<'py>,
    points: PyListVoxelPoint<'py>,
    voxel_map: &_VoxelHashMap,
    initial_guess: PyIsometryMatrix3<'py>,
    max_correspondance_distance: f64,
    kernel: f64,
) -> &'py PyArray2<f64> {
    let points = points
        .try_as_matrix::<Dyn, U3, Dyn, Dyn>()
        .unwrap()
        .clone_owned();
    let initial_guess = initial_guess.try_as_matrix::<U4, U4, Dyn, Dyn>().unwrap();
    let pose = voxel_map
        .0
        .register_frame(points, initial_guess, max_correspondance_distance, kernel)
        .into_matrix4();
    pose.to_pyarray(py)
}

/// AdaptiveThreshold bindings
#[pyclass]
struct _AdaptiveThreshold(AdaptiveThreshold);

#[pymethods]
impl _AdaptiveThreshold {
    #[new]
    fn new(initial_threshold: f64, min_motion_th: f64, max_range: f64) -> Self {
        Self(AdaptiveThreshold::new(
            initial_threshold,
            min_motion_th,
            max_range,
        ))
    }

    fn _compute_threshold(&mut self) -> f64 {
        self.0.compute_threshold()
    }

    fn _update_model_deviation(&mut self, model_deviation: PyIsometryMatrix3) {
        let model_deviation = model_deviation.try_as_matrix::<U4, U4, Dyn, Dyn>().unwrap();
        self.0.update_model_deviation(model_deviation)
    }
}

/// DeSkewScan
#[pyfunction]
fn _deskew_scan<'py>(
    py: Python<'py>,
    frame: PyListVoxelPoint<'py>,
    timestamps: Vec<f64>,
    start_pose: PyIsometryMatrix3<'py>,
    finish_pose: PyIsometryMatrix3<'py>,
) -> &'py PyArray2<f64> {
    let frame = frame.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
    let start_pose = start_pose.try_as_matrix::<U4, U4, Dyn, Dyn>().unwrap();
    let finish_pose = finish_pose.try_as_matrix::<U4, U4, Dyn, Dyn>().unwrap();
    let scanned: Vec<_> = deskew::scan(&frame, &timestamps, start_pose, finish_pose)
        .map(|point| point.transpose())
        .collect();
    MatrixXx3::from_rows(&scanned).to_pyarray(py)
}

// preprocessing modules

#[pyfunction]
fn _voxel_down_sample<'py>(
    py: Python<'py>,
    frame: PyListVoxelPoint<'py>,
    voxel_size: f64,
) -> &'py PyArray2<f64> {
    let frame = frame.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
    let downsampled: Vec<_> = preprocessing::voxel_downsample(&frame, voxel_size)
        .map(|point| point.transpose())
        .collect();
    MatrixXx3::from_rows(&downsampled).to_pyarray(py)
}

#[pyfunction]
fn _preprocess<'py>(
    py: Python<'py>,
    frame: PyListVoxelPoint<'py>,
    max_range: f64,
    min_range: f64,
) -> &'py PyArray2<f64> {
    let frame = frame.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
    let preprocessed = preprocessing::preprocess(&frame, min_range..max_range)
        .map(|point| point.transpose())
        .collect::<Vec<_>>();
    MatrixXx3::from_rows(&preprocessed).to_pyarray(py)
}

#[pyfunction]
fn _correct_kitti_scan<'py>(py: Python<'py>, frame: PyListVoxelPoint<'py>) -> &'py PyArray2<f64> {
    let frame = frame.try_as_matrix::<Dyn, U3, Dyn, Dyn>().unwrap();
    let corrected: Vec<_> = preprocessing::correct_kitti_scan(&frame)
        .map(|point| point.transpose())
        .collect();
    MatrixXx3::from_rows(&corrected).to_pyarray(py)
}

// Metrics

#[pyfunction]
fn _kitti_seq_error(
    gt_poses: Vec<IsometryArray3>,
    results_poses: Vec<IsometryArray3>,
) -> (f64, f64) {
    metrics::seq_error(&gt_poses, &results_poses)
}

#[pyfunction]
fn _absolute_trajectory_error(
    gt_poses: Vec<IsometryArray3>,
    results_poses: Vec<IsometryArray3>,
) -> (f64, f64) {
    metrics::absolute_trajectory_error(&gt_poses, &results_poses)
}

#[pymodule]
fn kiss_icp_pybind(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // optimize performance
    #[cfg(feature = "sas")]
    {
        use pyo3::exceptions::PyException;
        use sas::{Sas, SystemType};

        let args = Sas {
            system_type: SystemType::Python,
        };
        args.init().map_err(|error| {
            PyException::new_err(format!(
                "failed to init {name}: {error}",
                name = env!("CARGO_CRATE_NAME"),
            ))
        })?;
    }

    m.add_function(wrap_pyfunction!(_Vector3dVector, m)?)?;

    m.add_class::<_VoxelHashMap>()?;

    // Point Cloud registration
    m.add_function(wrap_pyfunction!(_register_point_cloud, m)?)?;

    // AdaptiveThreshold bindings
    m.add_class::<_AdaptiveThreshold>()?;

    // DeSkewScan
    m.add_function(wrap_pyfunction!(_deskew_scan, m)?)?;

    // preprocessing modules
    m.add_function(wrap_pyfunction!(_voxel_down_sample, m)?)?;
    m.add_function(wrap_pyfunction!(_preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(_correct_kitti_scan, m)?)?;

    // Metrics
    m.add_function(wrap_pyfunction!(_kitti_seq_error, m)?)?;
    m.add_function(wrap_pyfunction!(_absolute_trajectory_error, m)?)?;

    Ok(())
}
