use std::f64;

use itertools::Itertools;
use nalgebra::{Isometry3, Matrix3xX, Rotation3};

use crate::{ops::eigen::umeyama, types::IntoIsometry3};

#[allow(dead_code)]
struct Errors {
    first_frame: usize,
    r_err: f64,
    t_err: f64,
    len: f64,
    speed: f64,
}

fn trajectory_distances(poses: &[impl Copy + IntoIsometry3]) -> Vec<f64> {
    let mut dist = Vec::with_capacity(poses.len());
    dist.push(0.0);

    poses
        .iter()
        .map(IntoIsometry3::into_isometry3)
        .tuple_windows()
        .map(|(p1, p2)| (p1.translation.vector - p2.translation.vector).norm())
        .enumerate()
        .for_each(|(i, d)| dist.push(dist[i] + d));
    dist
}

#[inline]
fn last_frame_from_segment_length(dist: &[f64], first_frame: usize, len: f64) -> Option<usize> {
    (first_frame..dist.len()).find(|&i| dist[i] > dist[first_frame] + len)
}

#[inline]
fn rotation_error(pose_error: Isometry3<f64>) -> f64 {
    let mat = pose_error.rotation.to_rotation_matrix();
    let a = mat[(0, 0)];
    let b = mat[(1, 1)];
    let c = mat[(2, 2)];
    let d = 0.5 * (a + b + c - 1.0);
    d.min(1.0).max(-1.0).acos()
}

#[inline]
fn translation_error(pose_error: Isometry3<f64>) -> f64 {
    pose_error.translation.vector.norm()
}

fn calc_sequence_errors(
    poses_gt: &[impl Copy + IntoIsometry3],
    poses_result: &[impl Copy + IntoIsometry3],
) -> Vec<Errors> {
    // parameters
    // TODO: All this not so beatifull C++ functions are taken from kitti-dev-kit
    let lengths = [100f64, 200., 300., 400., 500., 600., 700., 800.];
    let step_size = 10; // every second

    // pre-compute distances (from ground truth as reference)
    let num_poses = poses_gt.len().min(poses_result.len());
    let dist = trajectory_distances(&poses_gt[..num_poses]);

    // for all start positions do
    (0..num_poses)
        .step_by(step_size)
        .flat_map(|first_frame| {
            // for all segment lengths do
            lengths
                .iter()
                .copied()
                // current length
                .filter_map(|len| {
                    // compute last frame
                    let last_frame = last_frame_from_segment_length(&dist, first_frame, len)?;

                    // compute rotational and translational errors
                    let pose_delta_gt = poses_gt[first_frame].into_isometry3().inverse()
                        * poses_gt[last_frame].into_isometry3();
                    let pose_delta_result = poses_result[first_frame].into_isometry3().inverse()
                        * poses_result[last_frame].into_isometry3();
                    let pose_error = pose_delta_result.inverse() * pose_delta_gt;
                    let r_err = rotation_error(pose_error);
                    let t_err = translation_error(pose_error);

                    // compute speed
                    let num_frames = (last_frame - first_frame + 1) as f64;
                    let speed = len / (0.1 * num_frames);

                    // write to file
                    Some(Errors {
                        first_frame,
                        r_err: r_err / len,
                        t_err: t_err / len,
                        len,
                        speed,
                    })
                })
                .collect_vec()
        })
        .collect()
}

pub fn seq_error(
    poses_gt: &[impl Copy + IntoIsometry3],
    poses_result: &[impl Copy + IntoIsometry3],
) -> (f64, f64) {
    let err = calc_sequence_errors(poses_gt, poses_result);
    if err.is_empty() {
        return (f64::INFINITY, f64::INFINITY);
    }
    let err_len = err.len() as f64;

    let (t_err, r_err) = err
        .into_iter()
        .map(|Errors { r_err, t_err, .. }| (t_err, r_err))
        .fold((0.0, 0.0), |(t_err_a, r_err_a), (t_err_b, r_err_b)| {
            (t_err_a + t_err_b, r_err_a + r_err_b)
        });

    let avg_trans_error = 100.0 * (t_err / err_len);
    let avg_rot_error = (r_err / err_len) / f64::consts::PI * 180.0;
    (avg_trans_error, avg_rot_error)
}

pub fn absolute_trajectory_error(
    poses_gt: &[impl Copy + IntoIsometry3],
    poses_result: &[impl Copy + IntoIsometry3],
) -> (f64, f64) {
    let num_poses = poses_gt.len().min(poses_result.len());
    let mut source = Matrix3xX::<f64>::zeros(num_poses);
    let mut target = Matrix3xX::<f64>::zeros(num_poses);

    // Align the two trajectories using SVD-ICP (Umeyama algorithm)
    for i in 0..num_poses {
        source
            .column_mut(i)
            .copy_from(&poses_result[i].into_isometry3().translation.vector);
        target
            .column_mut(i)
            .copy_from(&poses_gt[i].into_isometry3().translation.vector);
    }
    let t_align_trajectories = umeyama(&source, &target, false);

    // ATE computation
    let (ate_rot, ate_trans) = (0..num_poses)
        .map(|j| {
            // Apply alignement matrix
            let t_estimate = t_align_trajectories * poses_result[j].into_matrix4();
            let t_ground_truth = poses_gt[j].into_matrix4();

            // Compute error in translation and rotation matrix (ungly)
            let delta_r = t_ground_truth.fixed_view::<3, 3>(0, 0)
                * t_estimate.fixed_view::<3, 3>(0, 0).transpose();
            let delta_t = t_ground_truth.fixed_view::<3, 1>(0, 3)
                - delta_r * t_estimate.fixed_view::<3, 1>(0, 3);

            // Get angular error
            let theta = Rotation3::from_matrix(&delta_r).angle();

            // Sum of Squares
            let ate_rot = theta * theta;
            let ate_trans = delta_t.norm_squared();
            (ate_rot, ate_trans)
        })
        .fold(
            (0.0, 0.0),
            |(ate_rot_a, ate_trans_a), (ate_rot_b, ate_trans_b)| {
                (ate_rot_a + ate_rot_b, ate_trans_a + ate_trans_b)
            },
        );

    // Get the RMSE
    let num_poses = num_poses as f64;
    let ate_rot = (ate_rot / num_poses).sqrt();
    let ate_trans = (ate_trans / num_poses).sqrt();
    (ate_rot, ate_trans)
}
