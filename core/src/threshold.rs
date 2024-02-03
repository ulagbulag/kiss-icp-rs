use nalgebra::Isometry3;

use crate::types::IntoIsometry3;

pub struct AdaptiveThreshold {
    // configurable parameters
    initial_threshold: f64,
    min_motion_th: f64,
    max_range: f64,

    // Local cache for ccomputation
    model_error_sse2: f64,
    num_samples: u32,
    model_deviation: Isometry3<f64>,
}

impl AdaptiveThreshold {
    #[inline]
    pub fn new(initial_threshold: f64, min_motion_th: f64, max_range: f64) -> Self {
        Self {
            // configurable parameters
            initial_threshold,
            min_motion_th,
            max_range,
            // Local cache for ccomputation
            model_error_sse2: 0.0,
            num_samples: 0,
            model_deviation: Isometry3::identity(),
        }
    }

    #[inline]
    pub fn update_model_deviation(&mut self, current_deviation: impl IntoIsometry3) {
        self.model_deviation = current_deviation.into_isometry3()
    }

    /// Returns the KISS-ICP adaptive threshold used in registration
    #[inline]
    pub fn compute_threshold(&mut self) -> f64 {
        let model_error = compute_model_error(self.model_deviation, self.max_range);
        if model_error > self.min_motion_th {
            self.model_error_sse2 += model_error * model_error;
            self.num_samples += 1;
        }

        if self.num_samples < 1 {
            return self.initial_threshold;
        }
        (self.model_error_sse2 / self.num_samples as f64).sqrt()
    }
}

#[inline]
fn compute_model_error(model_deviation: Isometry3<f64>, max_range: f64) -> f64 {
    let theta = model_deviation.rotation.angle();
    let delta_rot = 2.0 * max_range * (theta / 2.0).sin();
    let delta_trans = model_deviation.translation.vector.norm();
    delta_trans + delta_rot
}
