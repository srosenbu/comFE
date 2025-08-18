use nalgebra::{coordinates::XYZWAB, Const, SMatrix, SVector, SVectorView, SVectorViewMut, Storage, Vector};
use crate::consts::*;

pub trait Mandel<const DIM: usize> {
    fn trace(&self) -> f64;

    fn vol_dev(
        &self,
    ) -> (f64, SVector<f64,DIM>);

    fn trace_dev(&self) -> (f64, SVector<f64,DIM>);

    fn i_1(&self) -> f64 {
        self.trace()
    }

    fn j_2(&self) -> f64 {
        let (_, dev) = self.trace_dev();
        0.5 * dev.norm_squared()
    }
    
    fn i1_j2_dev(&self) -> (f64, f64, SVector<f64,DIM>) {
        let (i_1, dev) = self.trace_dev();
        let j_2 = 0.5 * dev.norm_squared();
        (i_1, j_2, dev)
    }
    
    fn mises_norm(&self) -> f64 {
        let j_2 = self.j_2();
        (3.0 * j_2).sqrt()
    }
    fn elasticty(strain: Self, mu:f64, kappa:f64) ->SVector<f64,DIM>;
    fn elasticity_inv(stress: Self, mu:f64, kappa:f64) -> SVector<f64,DIM>;
}

pub trait MandelMut<const DIM: usize> : Mandel<DIM> {
    fn add_volumetric(
        &mut self,
        volumetric: f64,
    );
}

pub trait MandelView<'a, const DIM: usize> : Mandel<DIM> {
    fn from_array(slice: &'a [f64; DIM]) -> Self;
}
pub trait MandelViewMut<'a, const DIM: usize> : MandelMut<DIM> {
    fn from_array(slice: &'a mut [f64; DIM]) -> Self;
}

macro_rules! impl_mandel {
    ($type:ty, $dim:expr) => {
        impl Mandel<$dim> for $type {
            fn trace(&self) -> f64 {
                self.x + self.y + self.z
            }

            fn vol_dev(&self) -> (f64, SVector<f64,$dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (volumetric, deviatoric)
            }

            fn trace_dev(&self) -> (f64, SVector<f64,$dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (trace, deviatoric)
            }
            fn elasticty(strain: Self, mu:f64, kappa:f64) ->SVector<f64,$dim> {
                let (vol, dev) = strain.vol_dev();
                let mut stress = mu * dev;
                stress.add_volumetric(vol*kappa);
                stress
            }
            fn elasticity_inv(stress: Self, mu:f64, kappa:f64) -> SVector<f64,$dim> {
                let (vol, dev) = stress.vol_dev();
                let mut strain = (1.0/(2.0*mu)) * dev;
                strain.add_volumetric(vol*(1.0/kappa));
                strain
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
        unsafe{SVectorView::<'a, f64, 6>::from_slice_unchecked(slice, 0)}
    }
}
impl<'a> MandelViewMut<'a, 6> for SVectorViewMut<'a, f64, 6> {
    fn from_array(slice: &'a mut [f64; 6]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe{SVectorViewMut::<'a, f64, 6>::from_slice_unchecked(slice,0)}
    }
}
impl<'a> MandelView<'a, 4> for SVectorView<'a, f64, 4> {
    fn from_array(slice: &'a [f64; 4]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe{SVectorView::<'a, f64, 4>::from_slice_unchecked(slice, 0)}
    }
}
impl<'a> MandelViewMut<'a, 4> for SVectorViewMut<'a, f64, 4> {
    fn from_array(slice: &'a mut [f64; 4]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe{SVectorViewMut::<'a, f64, 4>::from_slice_unchecked(slice,0)}
    }
}

pub fn elastic_tangent<const N:usize>(mu:f64, kappa:f64) -> SMatrix<f64, N,N> {
    (2.0 * mu) * const { projection_dev::<N>() }
            + kappa * const { sym_id_outer_sym_id::<N>() }
}

pub fn elastic_tangent_inv<const N:usize>(mu:f64, kappa:f64) -> SMatrix<f64, N,N> {
  (1.0/(2.0*mu)) * const { projection_dev::<N>() }
            + (1.0/kappa) * const { sym_id_outer_sym_id::<N>() }
}

#[cfg(test)]
mod tests_mandel {
    use super::*;

    #[test]
    fn test_elasticity_tangent() {
        let mu = 1.2e+9;
        let kappa = 1.6e+9;
        let E = elastic_tangent::<6>(mu,kappa);
        let E_inv = elastic_tangent_inv::<6>(mu, kappa);
        let eye = SMatrix::<f64,6,6>::identity();
        println!("{}", E);

        println!("{}", E_inv);
        println!("{}", E.try_inverse().unwrap());
        println!("{}", E*E_inv);
        assert!((E*E_inv-eye).norm() < 1e-12 + 1e-12*E.norm())
    }

}

