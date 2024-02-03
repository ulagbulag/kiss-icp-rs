use std::{f64, ops::Range};

use nalgebra::{Dyn, Matrix, RawStorage, UnitQuaternion, UnitVector3, Vector3, U3};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    types::VoxelPoint,
    voxel_hash_map::{VoxelHashMap, VoxelHashMapArgs},
};

/// Crop the frame with max/min ranges
#[inline]
pub fn preprocess<S>(
    frame: &Matrix<f64, Dyn, U3, S>,
    range: Range<f64>,
) -> impl '_ + Iterator<Item = VoxelPoint>
where
    S: RawStorage<f64, Dyn, U3>,
{
    frame
        .row_iter()
        .map(|point| point.transpose())
        .filter(move |point| {
            let norm = point.norm();
            norm > range.start && norm < range.end
        })
}

/// This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
/// the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
/// Originally introduced the calibration factor)
pub fn correct_kitti_scan<S>(
    frame: &Matrix<f64, Dyn, U3, S>,
) -> impl '_ + ParallelIterator<Item = VoxelPoint>
where
    S: Send + Sync + RawStorage<f64, Dyn, U3>,
{
    const VERTICAL_ANGLE_OFFSET: f64 = (0.205 * f64::consts::PI) / 180.0;
    (0..frame.nrows())
        .into_par_iter()
        .map(|i| frame.row(i).transpose())
        .map(|point| {
            let rotation_vector =
                UnitVector3::new_normalize(point.cross(&Vector3::new(0.0, 0.0, 1.0)));
            UnitQuaternion::from_axis_angle(&rotation_vector, VERTICAL_ANGLE_OFFSET) * point
        })
}

/// Voxelize point cloud keeping the original coordinates
#[inline]
pub fn voxel_downsample<S>(
    frame: &Matrix<f64, Dyn, U3, S>,
    voxel_size: f64,
) -> impl Iterator<Item = VoxelPoint>
where
    S: RawStorage<f64, Dyn, U3>,
{
    let args = VoxelHashMapArgs {
        max_points_per_voxel: 1,
        ..Default::default()
    };

    let mut map = VoxelHashMap::new(args, voxel_size);
    map.add_points(frame);
    map.into_point_cloud()
}
