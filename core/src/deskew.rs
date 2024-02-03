use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{
    ops::{
        sophus::{Exp, Log},
        transform::Transform,
    },
    types::{IntoIsometry3, IntoVoxelPoint, VoxelPoint},
};

pub fn scan<'a, T>(
    frame: &'a [T],
    timestamps: &'a [f64],
    start_pose: impl IntoIsometry3,
    finish_pose: impl IntoIsometry3,
) -> impl 'a + Iterator<Item = VoxelPoint>
where
    T: Copy + IntoVoxelPoint,
    &'a [T]: 'a + IntoIterator<Item = &'a T>,
    // <&'a [T] as IntoParallelIterator>::Iter: IndexedParallelIterator,
{
    /// TODO(Nacho) Explain what is the very important meaning of this param
    const MID_POSE_TIMESTAMP: f64 = 0.5;

    let start_pose = start_pose.into_isometry3();
    let finish_pose = finish_pose.into_isometry3();
    let delta_pose = (start_pose.inverse() * finish_pose).log();

    frame.iter().zip(timestamps).map(move |(frame, timestamp)| {
        let motion = ((timestamp - MID_POSE_TIMESTAMP) * delta_pose).exp();

        let mut point = frame.into_voxel_point();
        point.transform_mut(motion);
        point
    })
}
