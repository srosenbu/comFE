use std::collections::HashMap;
use std::num::NonZeroUsize;

use crate::consts::*;
use crate::utils::{slice_as_chunks, slice_as_chunks_mut};
use konst::{const_eq, eq_str};
use nalgebra::{SMatrix, SVector, SVectorView, Scalar};
use phf::{Map, OrderedMap};

pub type ConstitutiveModelFn<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const HISTORY: usize,
    const PARAMETERS: usize,
> = fn(
    f64,
    f64,
    &[f64; STRESS_STRAIN],
    &mut [f64; STRESS_STRAIN],
    Option<&mut [f64; TANGENT]>,
    &mut [f64; HISTORY],
    &[f64; PARAMETERS],
);

#[repr(C)]
pub enum Dim {
    Scalar,
    Vector(usize),
    Matrix(usize),
    RotatableVector(usize),
    RotatableMatrix(usize),
}
impl Dim {
    pub const fn len(&self) -> usize {
        match self {
            Dim::Scalar => 1,
            Dim::Vector(size) => *size,
            Dim::RotatableVector(size) => *size,
            Dim::Matrix(size) => *size,
            Dim::RotatableMatrix(size) => *size,
        }
    }
}

pub trait ConstitutiveModel<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
{
    const HISTORY_MAP: [(&'static str, Dim); N_HISTORY];
    const PARAMETERS_MAP: [(&'static str, Dim); N_PARAMETERS];

    fn evaluate(
        time: f64,
        del_time: f64,
        del_strain: &[f64; STRESS_STRAIN],
        stress: &mut [f64; STRESS_STRAIN],
        tangent: Option<&mut [f64; TANGENT]>,
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
    T: ConstitutiveModel<{ STRESS_STRAIN }, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>() -> bool {
    let parameters = T::PARAMETERS_MAP;
    let history = T::HISTORY_MAP;
    let mut i: usize = 0;
    let mut size_parameters: usize = 0;
    let mut size_history: usize = 0;
    while i < N_HISTORY {
        size_history += history[i].1.len();
        i += 1;
    }
    i = 0;
    while i < N_PARAMETERS {
        size_parameters += parameters[i].1.len();
        i += 1;
    }
    const_eq!(size_parameters, PARAMETERS) && const_eq!(size_history, HISTORY)
}

pub const fn get_parameter_index<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    T: ConstitutiveModel<STRESS_STRAIN, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>(
    name: &'static str,
) -> (usize, usize)
where
    T: ConstitutiveModel<STRESS_STRAIN, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
{
    let parameters = T::PARAMETERS_MAP;
    let mut i: usize = 0;
    let mut start: usize = 0;
    while i < N_PARAMETERS {
        if const_eq!(parameters[i].0, name) {
            return (start, parameters[i].1.len());
        }
        start += parameters[i].1.len();
        i += 1;
    }
    panic!("Parameter not found in model");
}
pub const fn get_history_index<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    T: ConstitutiveModel<STRESS_STRAIN, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>(
    name: &'static str,
) -> (usize, usize)
where
    T: ConstitutiveModel<STRESS_STRAIN, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
{
    let history = T::HISTORY_MAP;
    let mut i: usize = 0;
    let mut start: usize = 0;
    while i < N_HISTORY {
        if const_eq!(history[i].0, name) {
            return (start, history[i].1.len());
        }
        start += history[i].1.len();
        i += 1;
    }
    panic!("History not found in model");
}

pub fn evaluate_model<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL,
>(
    time: f64,
    del_time: f64,
    del_strain: &[f64],
    stress: &mut [f64],
    tangent: Option<&mut [f64]>,
    history: &mut [f64],
    parameters: &[f64],
) where
    MODEL: ConstitutiveModel<STRESS_STRAIN, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
{
    let parameters: [f64; PARAMETERS] = parameters
        .try_into()
        .expect("Slice length does not match array length");

    let stress_len = stress.len() / STRESS_STRAIN;
    let strain_len = del_strain.len() / STRESS_STRAIN;
    let history_len = history.len() / HISTORY;
    let tangent_len = tangent
        .as_ref()
        .map(|t| t.len() / STRESS_STRAIN.pow(2))
        .unwrap_or(0);

    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent.is_none())
    );

    // This unsafe code is ok because the slices are guaranteed to be the correct length
    let stress_ = unsafe { slice_as_chunks_mut::<f64, STRESS_STRAIN>(stress).unwrap_unchecked() };
    let del_strain_ =
        unsafe { slice_as_chunks::<f64, STRESS_STRAIN>(del_strain).unwrap_unchecked() };
    let history_ = unsafe { slice_as_chunks_mut::<f64, HISTORY>(history).unwrap_unchecked() };
    let mut tangent_ =
        unsafe { tangent.map(|t| slice_as_chunks_mut::<f64, TANGENT>(t).unwrap_unchecked()) };

    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = del_strain_[i];
        let mut history_chunk = history_[i];
        let tangent_chunk: Option<&mut [f64; TANGENT]> = tangent_.as_mut().map(|t| &mut t[i]);
        MODEL::evaluate(
            time,
            del_time,
            &del_strain_chunk,
            &mut stress_chunk,
            tangent_chunk,
            &mut history_chunk,
            &parameters,
        );
    }
}
pub fn evaluate_model_fn<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const HISTORY: usize,
    const PARAMETERS: usize,
>(
    model: ConstitutiveModelFn<STRESS_STRAIN, TANGENT, HISTORY, PARAMETERS>,
    time: f64,
    del_time: f64,
    del_strain: &[f64],
    stress: &mut [f64],
    tangent: Option<&mut [f64]>,
    history: &mut [f64],
    parameters: &[f64],
) {
    let parameters: [f64; PARAMETERS] = parameters
        .try_into()
        .expect("Slice length does not match array length");

    let stress_len = stress.len() / STRESS_STRAIN;
    let strain_len = del_strain.len() / STRESS_STRAIN;
    let history_len = history.len() / HISTORY;
    let tangent_len = tangent
        .as_ref()
        .map(|t| t.len() / STRESS_STRAIN.pow(2))
        .unwrap_or(0);

    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent.is_none())
    );
    // This unsafe code is ok because the slices are guaranteed to be the correct length
    let stress_ = unsafe { slice_as_chunks_mut::<f64, STRESS_STRAIN>(stress).unwrap_unchecked() };
    let del_strain_ =
        unsafe { slice_as_chunks::<f64, STRESS_STRAIN>(del_strain).unwrap_unchecked() };
    let history_ = unsafe { slice_as_chunks_mut::<f64, HISTORY>(history).unwrap_unchecked() };
    let mut tangent_ =
        unsafe { tangent.map(|t| slice_as_chunks_mut::<f64, TANGENT>(t).unwrap_unchecked()) };

    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = del_strain_[i];
        let mut history_chunk = history_[i];
        let tangent_chunk: Option<&mut [f64; TANGENT]> = tangent_.as_mut().map(|t| &mut t[i]);

        model(
            time,
            del_time,
            &del_strain_chunk,
            &mut stress_chunk,
            tangent_chunk,
            &mut history_chunk,
            &parameters,
        );
    }
}
