use nalgebra::{
    ArrayStorage, Isometry3, Matrix, Matrix3, Quaternion, Translation3, UnitQuaternion, Vector3,
    Vector6, U3,
};

pub trait Exp<T> {
    fn exp(&self) -> Isometry3<f64>;
}

impl Exp<f64> for Vector6<f64> {
    #[inline]
    fn exp(&self) -> Isometry3<f64> {
        let t: Vector3<f64> = self.fixed_view::<3, 1>(0, 0).into();
        let omega: Vector3<f64> = self.fixed_view::<3, 1>(3, 0).into();

        let (rotation, theta) = omega.exp_and_theta();
        let v = left_jacobian(omega, theta);
        Isometry3::from_parts(Translation3::from(v * t), rotation)
    }
}

pub trait ExpAndTheta<T> {
    fn exp_and_theta(&self) -> (UnitQuaternion<T>, T);
}

impl ExpAndTheta<f64> for Vector3<f64> {
    #[inline]
    fn exp_and_theta(&self) -> (UnitQuaternion<f64>, f64) {
        let theta_sq = self.norm_squared();

        let (theta, imag_factor, real_factor) = if theta_sq < EPSILON * EPSILON {
            let theta = 0.0;
            let theta_po4 = theta_sq * theta_sq;
            let imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4;
            let real_factor = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4;
            (theta, imag_factor, real_factor)
        } else {
            let theta = theta_sq.sqrt();
            let half_theta = 0.5 * theta;
            let sin_half_theta = half_theta.sin();
            let imag_factor = sin_half_theta / theta;
            let real_factor = half_theta.cos();
            (theta, imag_factor, real_factor)
        };

        let q = UnitQuaternion::new_normalize(Quaternion::new(
            real_factor,
            imag_factor * self.x,
            imag_factor * self.y,
            imag_factor * self.z,
        ));
        (q, theta)
    }
}

pub trait Hat<T, D, S> {
    fn hat(&self) -> Matrix<T, D, D, S>;
}

impl Hat<f64, U3, ArrayStorage<f64, 3, 3>> for Vector3<f64> {
    #[inline]
    fn hat(&self) -> Matrix3<f64> {
        Matrix3::new(
            0.0, -self.z, self.y, // row 1
            self.z, 0.0, -self.x, // row 2
            -self.y, self.x, 0.0, // row 3
        )
    }
}

pub trait Log<T> {
    fn log(&self) -> Vector6<T>;
}

impl Log<f64> for Isometry3<f64> {
    #[inline]
    fn log(&self) -> Vector6<f64> {
        let (omega, theta) = self.rotation.log_and_theta();
        let v_inv = left_jacobian_inverse(omega, theta);
        let head = v_inv * self.translation.vector;
        let tail = omega;
        Vector6::new(head.x, head.y, head.z, tail.x, tail.y, tail.z)
    }
}

pub trait LogAndTheta<T> {
    fn log_and_theta(&self) -> (Vector3<T>, T);
}

impl LogAndTheta<f64> for UnitQuaternion<f64> {
    #[inline]
    fn log_and_theta(&self) -> (Vector3<f64>, f64) {
        let squared_n = self.vector().norm_squared();
        let w = self.w;

        /*
        Atan-based log thanks to

        C. Hertzberg et al.:
        "Integrating Generic Sensor Fusion Algorithms with Sound State
        Representation through Encapsulation of Manifolds"
        Information Fusion, 2011
        */

        let (two_atan_nbyw_by_n, theta) = if squared_n < EPSILON * EPSILON {
            let squared_w = w * w;
            let two_atan_nbyw_by_n = 2.0 / w - (2.0 / 3.0) * (squared_n) / (w * squared_w);
            let theta = 2.0 * squared_n / w;
            (two_atan_nbyw_by_n, theta)
        } else {
            let n = squared_n.sqrt();

            // w < 0 ==> cos(theta/2) < 0 ==> theta > pi
            //
            // By convention, the condition |theta| < pi is imposed by wrapping theta
            // to pi; The wrap operation can be folded inside evaluation of atan2
            //
            // theta - pi = atan(sin(theta - pi), cos(theta - pi))
            //            = atan(-sin(theta), -cos(theta))
            //
            let atan_nbyw = if w < 0.0 { (-n).atan2(-w) } else { n.atan2(w) };
            let two_atan_nbyw_by_n = 2.0 * atan_nbyw / n;
            let theta = two_atan_nbyw_by_n * n;
            (two_atan_nbyw_by_n, theta)
        };

        let tangent = two_atan_nbyw_by_n * self.vector();
        (tangent, theta)
    }
}

#[inline]
pub fn left_jacobian(omega: Vector3<f64>, theta: f64) -> Matrix3<f64> {
    let theta_sq = theta * theta;
    let omega = omega.hat();

    if theta_sq < EPSILON * EPSILON {
        Matrix3::identity() + 0.5 * omega
    } else {
        Matrix3::identity()
            + (1.0 - theta.cos()) / theta_sq * omega
            + (theta - theta.sin()) / (theta_sq * theta) * (omega * omega)
    }
}

#[inline]
pub fn left_jacobian_inverse(omega: Vector3<f64>, theta: f64) -> Matrix3<f64> {
    let theta_sq = theta * theta;
    let omega = omega.hat();

    if theta_sq < EPSILON * EPSILON {
        Matrix3::identity() - 0.5 * omega + (1. / 12.) * (omega * omega)
    } else {
        let half_theta = 0.5 * theta;

        Matrix3::identity() - 0.5 * omega
            + (1.0 - 0.5 * theta * half_theta.cos() / half_theta.sin()) / (theta * theta)
                * (omega * omega)
    }
}

const EPSILON: f64 = 1e-10;
