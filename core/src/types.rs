use nalgebra::{
    ArrayStorage, Isometry3, Matrix, Matrix4, Storage, Translation3, UnitQuaternion, Vector3, U4,
};

pub type VoxelPoint = Vector3<f64>;

pub type IsometryArray3 = [[f64; 4]; 4];

pub type IsometryMatrix3<S = ArrayStorage<f64, 4, 4>> = Matrix<f64, U4, U4, S>;

pub trait IntoIsometry3 {
    fn into_matrix4(self) -> IsometryMatrix3;

    fn into_isometry3(self) -> Isometry3<f64>;

    fn to_matrix4(&self) -> IsometryMatrix3;

    fn to_isometry3(&self) -> Isometry3<f64>;
}

impl<T> IntoIsometry3 for &T
where
    T: Copy + IntoIsometry3,
{
    #[inline]
    fn into_matrix4(self) -> IsometryMatrix3 {
        <T as IntoIsometry3>::into_matrix4(*self)
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        <T as IntoIsometry3>::into_isometry3(*self)
    }

    #[inline]
    fn to_matrix4(&self) -> IsometryMatrix3 {
        <T as IntoIsometry3>::to_matrix4(*self)
    }

    #[inline]
    fn to_isometry3(&self) -> Isometry3<f64> {
        <T as IntoIsometry3>::to_isometry3(*self)
    }
}

impl IntoIsometry3 for Isometry3<f64> {
    #[inline]
    fn into_matrix4(self) -> IsometryMatrix3 {
        self.into()
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        self
    }

    #[inline]
    fn to_matrix4(&self) -> IsometryMatrix3 {
        self.to_matrix()
    }

    #[inline]
    fn to_isometry3(&self) -> Isometry3<f64> {
        *self
    }
}

impl IntoIsometry3 for IsometryArray3 {
    #[inline]
    fn into_matrix4(self) -> IsometryMatrix3 {
        IsometryMatrix3::from(self).transpose()
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        let matrix = self.into_matrix4();
        matrix_to_isometry(matrix)
    }

    #[inline]
    fn to_matrix4(&self) -> IsometryMatrix3 {
        (*self).into_matrix4()
    }

    #[inline]
    fn to_isometry3(&self) -> Isometry3<f64> {
        let matrix = self.to_matrix4();
        matrix_to_isometry(matrix)
    }
}

impl<S> IntoIsometry3 for IsometryMatrix3<S>
where
    S: Storage<f64, U4, U4>,
{
    #[inline]
    fn into_matrix4(self) -> IsometryMatrix3 {
        self.into_owned()
    }

    #[inline]
    fn into_isometry3(self) -> Isometry3<f64> {
        let matrix = self.into_matrix4();
        matrix_to_isometry(matrix)
    }

    #[inline]
    fn to_matrix4(&self) -> IsometryMatrix3 {
        self.clone_owned()
    }

    #[inline]
    fn to_isometry3(&self) -> Isometry3<f64> {
        let matrix = self.to_matrix4();
        matrix_to_isometry(matrix)
    }
}

fn matrix_to_isometry(matrix: Matrix4<f64>) -> Isometry3<f64> {
    let rotation = UnitQuaternion::from_matrix(&matrix.fixed_view::<3, 3>(0, 0).into());
    let translation = Translation3 {
        vector: matrix.fixed_view::<3, 1>(0, 3).into(),
    };
    Isometry3::from_parts(translation, rotation)
}
