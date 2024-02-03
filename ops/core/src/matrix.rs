use std::{
    num::FpCategory,
    ops::{MulAssign, SubAssign},
};

use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, Matrix, RawStorageMut, Scalar, Storage, Vector, U1,
};

pub trait MatrixOps<T, R, C, S>
where
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn clip_normal(self) -> Self
    where
        S: RawStorageMut<f64, R, C>;
}

impl<R, C, S> MatrixOps<f64, R, C, S> for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
    DefaultAllocator: Allocator<f64, R, C>,
{
    #[inline]
    fn clip_normal(mut self) -> Self
    where
        S: RawStorageMut<f64, R, C>,
    {
        const MAX: f64 = 1e+6;
        const MIN: f64 = 1.0 / MAX;

        self.iter_mut().for_each(|value| {
            *value = match value.classify() {
                FpCategory::Nan | FpCategory::Subnormal | FpCategory::Zero => 0.0,
                FpCategory::Infinite => value.signum() * MAX,
                FpCategory::Normal => value.signum() * value.abs().max(MIN).min(MAX),
            }
        });
        self
    }
}

pub trait MatrixVectorOps<T, R, C, S>
where
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn vector_copy_from<S2>(&mut self, rhs: &Vector<T, R, S2>)
    where
        S: RawStorageMut<T, R, C>,
        S2: Storage<T, R, U1>;

    fn vector_sub<S2>(
        &self,
        rhs: &Vector<T, R, S2>,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        T: SubAssign,
        S2: Storage<T, R, U1>;

    fn vector_sub_assign<S2>(&mut self, rhs: &Vector<T, R, S2>)
    where
        T: SubAssign,
        S: RawStorageMut<T, R, C>,
        S2: Storage<T, R, U1>;

    fn vector_mul<S2>(
        &self,
        rhs: &Vector<T, R, S2>,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        T: MulAssign,
        S2: Storage<T, R, U1>;

    fn vector_mul_assign<S2>(&mut self, rhs: &Vector<T, R, S2>)
    where
        T: MulAssign,
        S: RawStorageMut<T, R, C>,
        S2: Storage<T, R, U1>;
}

impl<T, R, C, S> MatrixVectorOps<T, R, C, S> for Matrix<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<T, R, C>,
{
    #[inline]
    fn vector_copy_from<S2>(&mut self, rhs: &Vector<T, R, S2>)
    where
        S: RawStorageMut<T, R, C>,
        S2: Storage<T, R, U1>,
    {
        for i in 0..self.nrows() {
            let rhs = &rhs[i];
            for j in 0..self.ncols() {
                self[(i, j)] = rhs.clone();
            }
        }
    }

    #[inline]
    fn vector_sub<S2>(
        &self,
        rhs: &Vector<T, R, S2>,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        T: SubAssign,
        S2: Storage<T, R, U1>,
    {
        let mut res = self.clone_owned();
        res.vector_sub_assign(rhs);
        res
    }

    #[inline]
    fn vector_sub_assign<S2>(&mut self, rhs: &Vector<T, R, S2>)
    where
        T: SubAssign,
        S: RawStorageMut<T, R, C>,
        S2: Storage<T, R, U1>,
    {
        for i in 0..self.nrows() {
            let rhs = &rhs[i];
            for j in 0..self.ncols() {
                self[(i, j)] -= rhs.clone();
            }
        }
    }

    #[inline]
    fn vector_mul<S2>(
        &self,
        rhs: &Vector<T, R, S2>,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        T: MulAssign,
        S2: Storage<T, R, U1>,
    {
        let mut res = self.clone_owned();
        res.vector_mul_assign(rhs);
        res
    }

    #[inline]
    fn vector_mul_assign<S2>(&mut self, rhs: &Vector<T, R, S2>)
    where
        T: MulAssign,
        S: RawStorageMut<T, R, C>,
        S2: Storage<T, R, U1>,
    {
        for i in 0..self.nrows() {
            let rhs = &rhs[i];
            for j in 0..self.ncols() {
                self[(i, j)] *= rhs.clone();
            }
        }
    }
}
