use nalgebra::{ArrayStorage, Isometry3, Matrix4, Translation3, UnitQuaternion, Vector3};

pub type VoxelPoint = Vector3<f64>;

pub type PyIsometry3 = [[f64; 4]; 4];

pub trait IntoIsometry3 {
    fn into_matrix4(self) -> Matrix4<f64>;

    fn into_py_isometry3(self) -> PyIsometry3;

    fn into_isometry3(self) -> Isometry3<f64>;
}

impl<T> IntoIsometry3 for &T
where
    T: Copy + IntoIsometry3,
{
    #[inline]
    fn into_matrix4(self) -> Matrix4<f64> {
        <T as IntoIsometry3>::into_matrix4(*self)
    }

    #[inline]
    fn into_py_isometry3(self) -> PyIsometry3 {
        <T as IntoIsometry3>::into_py_isometry3(*self)
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        <T as IntoIsometry3>::into_isometry3(*self)
    }
}

impl IntoIsometry3 for Isometry3<f64> {
    #[inline]
    fn into_matrix4(self) -> Matrix4<f64> {
        self.to_matrix()
    }

    #[inline]
    fn into_py_isometry3(self) -> PyIsometry3 {
        self.into_matrix4().transpose().data.0
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        self
    }
}

impl IntoIsometry3 for PyIsometry3 {
    #[inline]
    fn into_matrix4(self) -> Matrix4<f64> {
        Matrix4::from(self).transpose()
    }

    #[inline]
    fn into_py_isometry3(self) -> PyIsometry3 {
        self
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        let matrix = self.into_matrix4();
        let rotation = UnitQuaternion::from_matrix(&matrix.fixed_view::<3, 3>(0, 0).into());
        let translation = Translation3 {
            vector: matrix.fixed_view::<3, 1>(0, 3).into(),
        };
        Isometry3::from_parts(translation, rotation)
    }
}

pub type PyVoxelPoint = [f64; 3];

pub trait IntoVoxelPoint {
    fn into_py_voxel_point(self) -> PyVoxelPoint;

    fn into_voxel_point(self) -> VoxelPoint;
}

impl<T> IntoVoxelPoint for &T
where
    T: Copy + IntoVoxelPoint,
{
    #[inline]
    fn into_py_voxel_point(self) -> PyVoxelPoint {
        <T as IntoVoxelPoint>::into_py_voxel_point(*self)
    }

    #[inline]
    fn into_voxel_point(self) -> VoxelPoint {
        <T as IntoVoxelPoint>::into_voxel_point(*self)
    }
}

impl IntoVoxelPoint for VoxelPoint {
    #[inline]
    fn into_py_voxel_point(self) -> PyVoxelPoint {
        self.data.0[0]
    }

    #[inline]
    fn into_voxel_point(self) -> VoxelPoint {
        self
    }
}

impl IntoVoxelPoint for PyVoxelPoint {
    #[inline]
    fn into_py_voxel_point(self) -> PyVoxelPoint {
        self
    }

    #[inline]
    fn into_voxel_point(self) -> VoxelPoint {
        VoxelPoint::from_data(ArrayStorage([self]))
    }
}
