use std::collections::HashMap;
use std::num::NonZeroUsize;

use crate::{
    consts::*,
    mandel::{MandelView, MandelViewMut},
};
use konst::{const_eq, eq_str};
use nalgebra::{SMatrix, SMatrixViewMut, SVector, SVectorView, SVectorViewMut, Scalar};
//use phf::{Map, OrderedMap};
//use serde::Serialize;

#[repr(C)]
#[derive(Copy, Clone)]
pub enum QDim {
    Scalar,
    Vector(usize),
    Matrix(usize),
    RotatableVector(usize),
    RotatableMatrix(usize),
}
impl QDim {
    pub const fn dim(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(size) => *size,
            QDim::RotatableVector(size) => *size,
            QDim::Matrix(size) => *size,
            QDim::RotatableMatrix(size) => *size,
        }
    }
    pub const fn size(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(size) => *size,
            QDim::RotatableVector(size) => *size,
            QDim::Matrix(size) => *size * *size,
            QDim::RotatableMatrix(size) => *size * *size,
        }
    }
}

#[macro_export]
macro_rules! q_dim_data_type {
    ((QDim::Scalar)) => {
        f64
    };
    ((QDim::Vector($size:expr))) => {
        SVector<f64, $size>
    };
    ((QDim::RotatableVector($size:expr))) => {
        SVector<f64, $size>
    };
    ((QDim::Matrix($size:expr))) => {
        SMatrix<f64, $size, $size>
    };
    ((QDim::RotatableMatrix($size:expr))) => {
        SMatrix<f64, $size, $size>
    };
}
pub trait ArrayEquivalent<const N: usize>: Sized {
    fn from_array(array: &[f64; N]) -> &Self;
    fn from_array_mut(array: &mut [f64; N]) -> &mut Self;

    fn as_array(&self) -> &[f64; N];
    fn as_array_mut(&mut self) -> &mut [f64; N];
}
pub trait StaticMap<const N: usize, T>
where
    T: Copy,
{
    const MAP: [(&'static str, T); N];

    //Todo: When constant traits become a thing, make this a constant function
    fn get(name: &str) -> Option<T> {
        for i in 0..N {
            if eq_str(Self::MAP[i].0, name) {
                return Some(Self::MAP[i].1);
            }
        }
        None
    }
    fn get_index(name: &str) -> Option<usize> {
        for i in 0..N {
            if eq_str(Self::MAP[i].0, name) {
                return Some(i);
            }
        }
        None
    }
}

impl<const N: usize> ArrayEquivalent<N> for [f64; N] {
    #[inline]
    fn from_array(array: &[f64; N]) -> &Self {
        array
    }

    #[inline]
    fn from_array_mut(array: &mut [f64; N]) -> &mut Self {
        array
    }

    #[inline]
    fn as_array(&self) -> &[f64; N] {
        self
    }

    #[inline]
    fn as_array_mut(&mut self) -> &mut [f64; N] {
        self
    }
}
impl<const N: usize> StaticMap<1, QDim> for [f64; N] {
    const MAP: [(&'static str, QDim); 1] = [("all_fields", QDim::Vector(N))];
}

#[macro_export]
macro_rules! create_history_parameter_struct {
    ($struct_name:ident, $n:expr, $size:expr, [$(($field_name:ident, $qdim:tt)),*]) => {
        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
        pub struct $struct_name {
            $(
                $field_name: q_dim_data_type!($qdim),
            )*
        }

        impl StaticMap<$n, QDim> for $struct_name {
            const MAP: [(&'static str, QDim); $n] = [
                $((stringify!($field_name), $qdim)),*
            ];
        }
        impl_array_equivalent!($struct_name, $size);
    };
}

#[macro_export]
macro_rules! impl_array_equivalent {
    ($type:ty, $size:expr) => {
        const _: () = assert!(
            std::mem::size_of::<[f64; $size]>() == std::mem::size_of::<$type>(),
            "size mismatch"
        );

        impl ArrayEquivalent<$size> for $type {
            #[inline]
            fn from_array(array: &[f64; $size]) -> &Self {
                unsafe { &*(array as *const [f64; $size] as *const Self) }
            }

            #[inline]
            fn from_array_mut(array: &mut [f64; $size]) -> &mut Self {
                unsafe { &mut *(array as *mut [f64; $size] as *mut Self) }
            }

            #[inline]
            fn as_array(&self) -> &[f64; $size] {
                unsafe { &*(self as *const Self as *const [f64; $size]) }
            }

            #[inline]
            fn as_array_mut(&mut self) -> &mut [f64; $size] {
                unsafe { &mut *(self as *mut Self as *mut [f64; $size]) }
            }
        }
    };
}
pub trait ConstitutiveModel<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
where
    Self: Sized,
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim>;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn new(parameters: &Self::Parameters) -> Self;

    fn evaluate(
        &self,
        time: f64,
        del_time: f64,
        strain: &[f64; STRESS_STRAIN],
        del_strain: &[f64; STRESS_STRAIN],
        stress: &mut [f64; STRESS_STRAIN],
        tangent: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]>,
        history: &mut [f64; HISTORY],
        parameters: &[f64; PARAMETERS],
    );
}
pub trait ConstitutiveModelFn<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim>;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn evaluate(
        time: f64,
        del_time: f64,
        strain: &[f64; STRESS_STRAIN],
        del_strain: &[f64; STRESS_STRAIN],
        stress: &mut [f64; STRESS_STRAIN],
        tangent: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]>,
        history: &mut [f64; HISTORY],
        parameters: &[f64; PARAMETERS],
    );
}
pub trait ConstitutiveModelFunction<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim> + Copy;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim> + Copy;

    fn evaluate<const TANGENT: bool>(
        time: f64,
        del_time: f64,
        strain: SVector<f64, STRESS_STRAIN>,
        del_strain: SVector<f64, STRESS_STRAIN>,
        stress: SVector<f64, STRESS_STRAIN>,
        history: Self::History,
        parameters: Self::Parameters,
    ) -> (
        SVector<f64, STRESS_STRAIN>,
        Option<SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>>,
        Self::History,
    );
}

pub fn input_as_nalgebra<
    'a,
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>(
    strain: &'a [f64; STRESS_STRAIN],
    del_strain: &'a [f64; STRESS_STRAIN],
    stress: &'a mut [f64; STRESS_STRAIN],
    tangent: Option<&'a mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]>,
) -> (
    SVectorView<'a, f64, STRESS_STRAIN>,
    SVectorView<'a, f64, STRESS_STRAIN>,
    SVectorViewMut<'a, f64, STRESS_STRAIN>,
    Option<SMatrixViewMut<'a, f64, STRESS_STRAIN, STRESS_STRAIN>>,
)
where
    SVectorView<'a, f64, STRESS_STRAIN>: MandelView<'a, STRESS_STRAIN>,
    SVectorViewMut<'a, f64, STRESS_STRAIN>: MandelViewMut<'a, STRESS_STRAIN>,
{
    let strain_view = SVectorView::from_array(strain);
    let del_strain_view = SVectorView::from_array(del_strain);
    let stress_view_mut = SVectorViewMut::from_array(stress);

    let tangent_view_mut = tangent.map(|t| SMatrixViewMut::from_slice(t.as_flattened_mut()));

    (
        strain_view,
        del_strain_view,
        stress_view_mut,
        tangent_view_mut,
    )
}

pub trait ConstitutiveModelFn3D<
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim>;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn evaluate(
        time: f64,
        del_time: f64,
        strain: &[f64; 6],
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; HISTORY],
        parameters: &[f64; PARAMETERS],
    );
}
pub trait ConstitutiveModelFnPlaneStrain<
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim>;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn evaluate(
        time: f64,
        del_time: f64,
        strain: &[f64; 4],
        del_strain: &[f64; 4],
        stress: &mut [f64; 4],
        tangent: Option<&mut [[f64; 4]; 4]>,
        history: &mut [f64; HISTORY],
        parameters: &[f64; PARAMETERS],
    );
}

pub const fn check_constitutive_model_maps<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    T: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>() -> bool {
    let parameters = T::Parameters::MAP;
    let history = T::History::MAP;
    let mut i: usize = 0;
    let mut size_parameters: usize = 0;
    let mut size_history: usize = 0;
    while i < N_HISTORY {
        size_history += history[i].1.size();
        i += 1;
    }
    i = 0;
    while i < N_PARAMETERS {
        size_parameters += parameters[i].1.size();
        i += 1;
    }
    const_eq!(size_parameters, PARAMETERS) && const_eq!(size_history, HISTORY)
}

pub fn evaluate_model<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS> + Sized,
>(
    time: f64,
    del_time: f64,
    strain: &[f64],
    del_strain: &[f64],
    stress: &mut [f64],
    tangent: Option<&mut [f64]>,
    history: &mut [f64],
    parameters: &[f64],
) {
    let parameters: [f64; PARAMETERS] = parameters
        .try_into()
        .expect(&format!("Length of parameters slice does not match the expected length. Expected: {}, got: {}.",
            PARAMETERS, parameters.len()));

    let (stress_, stress_rest) = stress.as_chunks_mut::<STRESS_STRAIN>();
    let (strain_, strain_rest) = strain.as_chunks::<STRESS_STRAIN>();
    let (del_strain_, del_strain_rest) = del_strain.as_chunks::<STRESS_STRAIN>();
    let (history_, history_rest) = history.as_chunks_mut::<HISTORY>();
    let mut tangent_ = {
        match tangent {
            Some(t) => {
                let (tangent_, tangent_rest_1) = t.as_chunks_mut::<STRESS_STRAIN>();
                let (tangent_, tangent_rest_2) = tangent_.as_chunks_mut::<STRESS_STRAIN>();
                Some(tangent_)
            }
            None => None,
        }
    };

    let stress_len = stress_.len();
    let del_strain_len = del_strain_.len();
    let strain_len = strain_.len();
    let history_len = history_.len();
    let tangent_len = tangent_.as_ref().map_or(0, |t| t.len());

    assert!(
        stress_len == del_strain_len
            && stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent_.is_none()),
        "Stress, strain, history, and tangent lengths do not match: \
        stress_len: {}, strain_len: {}, del_strain_len: {}, history_len: {}, tangent_len: {}",
        stress_len,
        strain_len,
        del_strain_len,
        history_len,
        tangent_len
    );
    assert!(
        stress_rest.is_empty()
            && del_strain_rest.is_empty()
            && history_rest.is_empty()
            && tangent_.as_ref().map_or(true, |t| t.is_empty()),
        "Input slices are not of the correct length: \
        stress_rest: {:?}, del_strain_rest: {:?}, history_rest: {:?}",
        stress_rest,
        del_strain_rest,
        history_rest
    );

    for i in 0..stress_len {
        let tangent_chunk: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]> =
            tangent_.as_mut().map(|t| &mut t[i]);

        MODEL::evaluate(
            time,
            del_time,
            &strain_[i],
            &del_strain_[i],
            &mut stress_[i],
            tangent_chunk,
            &mut history_[i],
            &parameters,
        );
    }
}
