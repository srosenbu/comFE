use crate::consts::*;
use nalgebra::{
    Const, SMatrix, SVector, SVectorView, SVectorViewMut, Storage, Vector, coordinates::XYZWAB,
};

pub trait Mandel<const DIM: usize> {
    fn trace(&self) -> f64;

    fn vol_dev(&self) -> (f64, SVector<f64, DIM>);

    fn trace_dev(&self) -> (f64, SVector<f64, DIM>);

    fn i_1(&self) -> f64 {
        self.trace()
    }

    fn j_2(&self) -> f64 {
        let (_, dev) = self.trace_dev();
        0.5 * dev.norm_squared()
    }

    fn i1_j2_dev(&self) -> (f64, f64, SVector<f64, DIM>) {
        let (i_1, dev) = self.trace_dev();
        let j_2 = 0.5 * dev.norm_squared();
        (i_1, j_2, dev)
    }

    fn mises_norm(&self) -> f64 {
        let j_2 = self.j_2();
        (3.0 * j_2).sqrt()
    }
}

pub trait MandelMut<const DIM: usize>: Mandel<DIM> {
    fn add_volumetric(&mut self, volumetric: f64);
}

pub trait MandelView<'a, const DIM: usize>: Mandel<DIM> {
    fn from_array(slice: &'a [f64; DIM]) -> Self;
}
pub trait MandelViewMut<'a, const DIM: usize>: MandelMut<DIM> {
    fn from_array(slice: &'a mut [f64; DIM]) -> Self;
}

macro_rules! impl_mandel {
    ($type:ty, $dim:expr) => {
        impl Mandel<$dim> for $type {
            fn trace(&self) -> f64 {
                self.x + self.y + self.z
            }

            fn vol_dev(&self) -> (f64, SVector<f64, $dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (volumetric, deviatoric)
            }

            fn trace_dev(&self) -> (f64, SVector<f64, $dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (trace, deviatoric)
            }
        }
    };
}
macro_rules! impl_mandel_mut {
    ($type:ty, $dim:expr) => {
        impl MandelMut<$dim> for $type {
            fn add_volumetric(&mut self, volumetric: f64) {
                self.x += volumetric;
                self.y += volumetric;
                self.z += volumetric;
            }
        }
    };
}
// Use the macro to implement Mandel for SVector<f64, 6>
impl_mandel_mut!(SVector<f64, 6>, 6);
impl_mandel!(SVector<f64, 6>, 6);
impl_mandel_mut!(SVector<f64, 4>, 4);
impl_mandel!(SVector<f64, 4>, 4);

impl_mandel!(SVectorView<'_, f64, 6>, 6);
impl_mandel!(SVectorView<'_, f64, 4>, 4);

impl_mandel_mut!(SVectorViewMut<'_, f64, 6>, 6);
impl_mandel!(SVectorViewMut<'_, f64, 6>, 6);
impl_mandel_mut!(SVectorViewMut<'_, f64, 4>, 4);
impl_mandel!(SVectorViewMut<'_, f64, 4>, 4);

impl<'a> MandelView<'a, 6> for SVectorView<'a, f64, 6> {
    fn from_array(slice: &'a [f64; 6]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorView::<'a, f64, 6>::from_slice_unchecked(slice, 0) }
    }
}
impl<'a> MandelViewMut<'a, 6> for SVectorViewMut<'a, f64, 6> {
    fn from_array(slice: &'a mut [f64; 6]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorViewMut::<'a, f64, 6>::from_slice_unchecked(slice, 0) }
    }
}
impl<'a> MandelView<'a, 4> for SVectorView<'a, f64, 4> {
    fn from_array(slice: &'a [f64; 4]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorView::<'a, f64, 4>::from_slice_unchecked(slice, 0) }
    }
}
impl<'a> MandelViewMut<'a, 4> for SVectorViewMut<'a, f64, 4> {
    fn from_array(slice: &'a mut [f64; 4]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorViewMut::<'a, f64, 4>::from_slice_unchecked(slice, 0) }
    }
}


/// Determines the elastic tangent matrix in Mandel notation. 
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
pub fn isotropic_elastic_tangent<const N: usize>(mu: f64, kappa: f64) -> SMatrix<f64, N, N> {
    (2.0 * mu) * const { projection_dev::<N>() } + (3.0 * kappa) * const { projection_vol::<N>() }
}

/// Determines the inverse of the elastic tangent matrix in Mandel notation. It does so
/// by calling `isotropic_elastic_tangent` with $\frac{1}{4\mu}$ and $\frac{1}{9\kappa}$
/// to avoid calculating the inverse directly. The creation is therefore basically as fast as
/// the creation of the tangent matrix. 
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
pub fn isotropic_elastic_tangent_inv<const N: usize>(mu: f64, kappa: f64) -> SMatrix<f64, N, N> {
    let mu_inv = 1.0 / (4.0 * mu);
    let kappa_inv = 1.0 / (9.0 * kappa);
    isotropic_elastic_tangent(mu_inv, kappa_inv)
}

#[cfg(test)]
mod tests_mandel {
    use nalgebra::ArrayStorage;

    use super::*;

    static MU: f64 = 1.2e+9;
    static KAPPA: f64 = 1.6e+9;
    static LAMBDA: f64 = KAPPA - 2.0 * MU / 3.0;
    static TANGENT_ANALYTICAL: SMatrix<f64, 6, 6> =
        SMatrix::<f64, 6, 6>::from_array_storage(ArrayStorage([
            [2. * MU + LAMBDA, LAMBDA, LAMBDA, 0.0, 0.0, 0.0],
            [LAMBDA, 2. * MU + LAMBDA, LAMBDA, 0.0, 0.0, 0.0],
            [LAMBDA, LAMBDA, 2. * MU + LAMBDA, 0.0, 0.0, 0.0],
            [0., 0., 0., 2. * MU, 0., 0.],
            [0., 0., 0., 0., 2. * MU, 0.],
            [0., 0., 0., 0., 0., 2. * MU],
        ])); //Note that the memory layout is column wise, but the matrix is symmetric, so it does not matter

    #[test]
    fn test_tangent() {
        //test that the generator function equals the tangent from wikipedia
        assert!(
            (isotropic_elastic_tangent(MU, KAPPA) - TANGENT_ANALYTICAL).norm()
                < 1e-14 + 1e-14 * TANGENT_ANALYTICAL.norm()
        )
    }
    #[test]
    fn test_elasticity_tangent_inverse() {
        //test that the inverse tangent generator function actually produces an inverse
        let tangent = isotropic_elastic_tangent::<6>(MU, KAPPA);
        let tangent_inv = isotropic_elastic_tangent_inv::<6>(MU, KAPPA);
        assert!((tangent * tangent_inv - SMatrix::<f64, 6, 6>::identity()).norm() < 1e-14)
    }
}
