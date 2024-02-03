use nalgebra::{Dyn, Isometry3, Matrix, Point3, RawStorageMut, Vector3, U3};

pub trait Transform {
    fn transform_mut(&mut self, transform: Isometry3<f64>);
}

impl Transform for Isometry3<f64> {
    #[inline]
    fn transform_mut(&mut self, transform: Isometry3<f64>) {
        *self = transform * *self
    }
}

impl Transform for Vector3<f64> {
    #[inline]
    fn transform_mut(&mut self, transform: Isometry3<f64>) {
        let t = Point3::from(*self);
        self.copy_from(&(transform * t).coords)
    }
}

impl<S> Transform for Matrix<f64, Dyn, U3, S>
where
    S: RawStorageMut<f64, Dyn, U3>,
{
    #[inline]
    fn transform_mut(&mut self, transform: Isometry3<f64>) {
        self.row_iter_mut().for_each(|mut point| {
            point.copy_from(&{
                let mut point = point.transpose();
                point.transform_mut(transform);
                point.transpose()
            })
        })
    }
}

impl Transform for Vec<Vector3<f64>> {
    #[inline]
    fn transform_mut(&mut self, transform: Isometry3<f64>) {
        self.iter_mut()
            .for_each(|point| point.transform_mut(transform))
    }
}
