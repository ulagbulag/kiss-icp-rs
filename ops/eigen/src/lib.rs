use std::ops::MulAssign;

use kiss_icp_ops_core::matrix::MatrixVectorOps;
use nalgebra::{Matrix3xX, Matrix4, Vector3};
use nalgebra_lapack::SVD;

pub fn umeyama(src: &Matrix3xX<f64>, dst: &Matrix3xX<f64>, with_scaling: bool) -> Matrix4<f64> {
    let n = src.ncols(); // number of measurements

    // required for demeaning ...
    let one_over_n = 1.0 / n as f64;

    // computation of mean
    let src_mean = src.column_mean();
    let dst_mean = dst.column_mean();

    // demeaning of src and dst points
    let src_demean = src.vector_sub(&src_mean);
    let dst_demean = dst.vector_sub(&dst_mean);

    // Eq. (36)-(37)
    let src_var = src_demean
        .row_iter()
        .map(|column| column.norm_squared())
        .sum::<f64>()
        * one_over_n;

    // Eq. (38)
    let sigma = one_over_n * dst_demean * src_demean.transpose();

    let SVD {
        u,
        vt: v_t,
        singular_values,
    } = SVD::new(sigma).unwrap();

    // Initialize the resulting transformation with an identity matrix...
    let mut rt = Matrix4::identity();

    // Eq. (39)
    let mut s = Vector3::<f64>::from_element(1.0);

    if u.determinant() * v_t.transpose().determinant() < 0.0 {
        s[2] = -1.0;
    }

    // Eq. (40) and (43)
    let rot = u.vector_mul(&s) * v_t;
    rt.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);

    let mut block = rt.fixed_view_mut::<3, 1>(0, 3);
    block.copy_from(&dst_mean);

    if with_scaling {
        // Eq. (42)
        let c = 1.0 / src_var * singular_values.dot(&s);

        // Eq. (41)
        block -= c * rot * src_mean;
        rt.fixed_view_mut::<3, 3>(0, 0).mul_assign(c);
    } else {
        block -= rot * src_mean;
    }

    rt
}
