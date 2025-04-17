use nalgebra::{coordinates::XYZWAB, Const, SMatrix, SVector, SVectorView, SVectorViewMut, Storage, Vector};
use crate::consts::{ID_6, SYM_ID_6, SYM_ID_6_OUTER_SYM_ID_6};

pub fn trace(vector: &XYZWAB<f64>) -> f64 {
    vector.x + vector.y + vector.z
}


pub trait Mandel<const DIM: usize> {
    fn trace(&self) -> f64;

    fn vol_dev(
        &self,
    ) -> (f64, SVector<f64, DIM>);

    fn trace_dev(&self) -> (f64, SVector<f64, DIM>);

    fn I_1(&self) -> f64 {
        self.trace()
    }

    fn J_2(&self) -> f64 {
        let (_, deviatoric) = self.vol_dev();
        0.5 * deviatoric.dot(&deviatoric)
    }
    
    fn mises_norm(&self) -> f64 {
        let j_2 = self.J_2();
        (3.0 * j_2).sqrt()
    }
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

impl_mandel!(SVectorView<'_, f64, 6>, 6);

impl_mandel_mut!(SVectorViewMut<'_, f64, 6>, 6);
impl_mandel!(SVectorViewMut<'_, f64, 6>, 6);


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
