use nalgebra::{Dyn, Matrix, RawStorage, U3};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{
    ops::{
        sophus::{Exp, Log},
        transform::Transform,
    },
    types::{IntoIsometry3, VoxelPoint},
};

pub fn scan<'a, S>(
    frame: &'a Matrix<f64, Dyn, U3, S>,
    timestamps: &'a [f64],
    start_pose: impl IntoIsometry3,
    finish_pose: impl IntoIsometry3,
) -> impl 'a + ParallelIterator<Item = VoxelPoint>
where
    S: Send + Sync + RawStorage<f64, Dyn, U3>,
{
    /// TODO(Nacho) Explain what is the very important meaning of this param
    const MID_POSE_TIMESTAMP: f64 = 0.5;

    let start_pose = start_pose.into_isometry3();
    let finish_pose = finish_pose.into_isometry3();
    let delta_pose = (start_pose.inverse() * finish_pose).log();

    let motions = timestamps
        .par_iter()
        .map(move |timestamp| ((timestamp - MID_POSE_TIMESTAMP) * delta_pose).exp());

    (0..frame.nrows())
        .into_par_iter()
        .map(|i| frame.row(i).transpose())
        .zip(motions)
        .map(move |(mut point, motion)| {
            point.transform_mut(motion);
            point
        })
}
