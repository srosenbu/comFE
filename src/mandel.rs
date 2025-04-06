use nalgebra::{coordinates::XYZWAB, Const, SMatrix, SVector, SVectorView, SVectorViewMut, Storage, Vector};
use crate::consts::{ID_6, SYM_ID_6, SYM_ID_6_OUTER_SYM_ID_6};

pub fn trace(vector: &XYZWAB<f64>) -> f64 {
    vector.x + vector.y + vector.z
}


pub trait Mandel<const DIM: usize> {
    fn trace(&self) -> f64;
    fn add_volumetric(
        &mut self,
        volumetric: f64,
    );
    fn vol_dev(
        &self,
    ) -> (f64, SVector<f64, DIM>);
}

pub trait MandelView<const DIM: usize> : Mandel<DIM> {
    fn from_array<'a>(slice: &'a [f64; DIM]) -> Self;
}
pub trait MandelViewMut<const DIM: usize> : Mandel<DIM> {
    fn from_array(slice: &mut [f64; DIM]) -> Self;
}

macro_rules! impl_mandel {
    ($type:ty, $dim:expr) => {
        impl Mandel<$dim> for $type {
            fn trace(&self) -> f64 {
                self.x + self.y + self.z
            }

            fn add_volumetric(&mut self, volumetric: f64) {
                self.x += volumetric;
                self.y += volumetric;
                self.z += volumetric;
            }

            fn vol_dev(&self) -> (f64, SVector<f64, $dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (volumetric, deviatoric)
            }
        }
    };
}

// Use the macro to implement Mandel for SVector<f64, 6>
impl_mandel!(SVector<f64, 6>, 6);
//impl_mandel!(SVectorView<'_, f64, 6>, 6);
impl_mandel!(SVectorViewMut<'_, f64, 6>, 6);

impl<'b> MandelView<6> for SVectorView<'b, f64, 6> {
    fn from_array<'a>(slice: &'a [f64; 6]) -> SVectorView<'a, f64, 6> {
        SVectorView::<'a, f64, 6>::from_slice(slice)
    }
}
impl MandelViewMut<6> for SVectorViewMut<f64, 6> {
    fn from_array(slice: &mut [f64; 6]) -> Self {
        SVectorViewMut::from_row_slice(slice)
    }
}
