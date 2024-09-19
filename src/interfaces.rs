use std::collections::HashMap;

use nalgebra::{SMatrix, SVector, SVectorView};

use crate::consts::*;
use crate::utils::{slice_as_chunks, slice_as_chunks_mut};

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
    Vector,
    Matrix,
    RotatableVector,
    RotatableMatrix,
}
pub trait ConstitutiveModel<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const HISTORY: usize,
    const PARAMETERS: usize,
>
{
    fn evaluate(
        time: f64,
        del_time: f64,
        del_strain: &[f64; STRESS_STRAIN],
        stress: &mut [f64; STRESS_STRAIN],
        tangent: Option<&mut [f64; TANGENT]>,
        history: &mut [f64; HISTORY],
        parameters: &[f64; PARAMETERS],
    );

    fn parameters() -> [String; PARAMETERS];

}

pub fn evaluate_model<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const HISTORY: usize,
    const PARAMETERS: usize,
    MODEL: ConstitutiveModel<STRESS_STRAIN, TANGENT, HISTORY, PARAMETERS>,
>(
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
    let tangent_len = tangent.map(|t| t.len() / STRESS_STRAIN.pow(2)).unwrap_or(0);


    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent.is_none())
    );
    // This unsafe code is ok because the slices are guaranteed to be the correct length
    let stress_ = slice_as_chunks_mut::<f64, STRESS_STRAIN>(stress).unwrap();
    let del_strain_ = slice_as_chunks::<f64, STRESS_STRAIN>(del_strain).unwrap();
    let history_ = slice_as_chunks_mut::<f64, HISTORY>(history).unwrap();
    let tangent_ = tangent.map(|t| slice_as_chunks_mut::<f64, TANGENT>(t).unwrap());

    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = del_strain_[i];
        let mut history_chunk = history_[i];
        let mut tangent_chunk: Option<&mut [f64; TANGENT]> = tangent_.map(|t| &mut t[i]); 
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
    let tangent_len = tangent.map(|t| t.len() / STRESS_STRAIN.pow(2)).unwrap_or(0);


    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent.is_none())
    );
    // This unsafe code is ok because the slices are guaranteed to be the correct length
    let stress_ = slice_as_chunks_mut::<f64, STRESS_STRAIN>(stress).unwrap();
    let del_strain_ = slice_as_chunks::<f64, STRESS_STRAIN>(del_strain).unwrap();
    let history_ = slice_as_chunks_mut::<f64, HISTORY>(history).unwrap();
    let tangent_ = tangent.map(|t| slice_as_chunks_mut::<f64, TANGENT>(t).unwrap());

    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = del_strain_[i];
        let mut history_chunk = history_[i];
        let mut tangent_chunk: Option<&mut [f64; TANGENT]> = tangent_.map(|t| &mut t[i]); 

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
