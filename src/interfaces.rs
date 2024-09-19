use std::collections::HashMap;

use nalgebra::{SMatrix, SVector, SVectorView};

use crate::consts::*;

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
    //fn check_dimensions() -> bool {
    //    let check = STRESS_STRAIN.pow(2) == TANGENT;
    //    let history_dim = Self::history().iter().fold(0, |acc, (name, dim, size)| {
    //        acc + size
    //    });
    //    let check = check && history_dim == HISTORY;
    //    check
    //}

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
    
    fn history() -> [(String, Dim, usize)];
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
    let tangent_len = match tangent {
        Some(t) => t.len() / STRESS_STRAIN.pow(2),
        None => 0
    };

    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent.is_none())
    );
    // This unsafe code is ok because the slices are guaranteed to be the correct length
    let stress_ = unsafe {
        std::slice::from_raw_parts_mut(
            stress.as_mut_ptr() as *mut [f64; STRESS_STRAIN],
            stress_len,
        )
    };
    let strain_ = unsafe {
        std::slice::from_raw_parts(
            del_strain.as_ptr() as *const [f64; STRESS_STRAIN],
            strain_len,
        )
    };
    let history_ = unsafe {
        std::slice::from_raw_parts_mut(
            history.as_mut_ptr() as *mut [f64; HISTORY],
            history_len,
        )
    };
    let tangent_ = {
    match tangent {
        Some(t) => unsafe {
            Some(std::slice::from_raw_parts_mut(
                t.as_mut_ptr() as *mut [f64; TANGENT],
                tangent_len,
            ))
        },
        None => None
    }
    };    
    
    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = strain_[i];
        let mut history_chunk = history_[i];
        let mut tangent_chunk:Option<&mut [f64; TANGENT]> = match tangent_ {
            Some(t) => Some(&mut t[i]),
            None => None
        };
        
        MODEL::evaluate(time, del_time, &del_strain_chunk, &mut stress_chunk, tangent_chunk, &mut history_chunk, &parameters);
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
    let tangent_len = match tangent {
        Some(t) => t.len() / STRESS_STRAIN.pow(2),
        None => 0
    };

    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent.is_none())
    );
    // This unsafe code is ok because the slices are guaranteed to be the correct length
    let stress_ = unsafe {
        std::slice::from_raw_parts_mut(
            stress.as_mut_ptr() as *mut [f64; STRESS_STRAIN],
            stress_len,
        )
    };
    let strain_ = unsafe {
        std::slice::from_raw_parts(
            del_strain.as_ptr() as *const [f64; STRESS_STRAIN],
            strain_len,
        )
    };
    let history_ = unsafe {
        std::slice::from_raw_parts_mut(
            history.as_mut_ptr() as *mut [f64; HISTORY],
            history_len,
        )
    };
    let tangent_ = {
    match tangent {
        Some(t) => unsafe {
            Some(std::slice::from_raw_parts_mut(
                t.as_mut_ptr() as *mut [f64; TANGENT],
                tangent_len,
            ))
        },
        None => None
    }
    };    
    
    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = strain_[i];
        let mut history_chunk = history_[i];
        let mut tangent_chunk:Option<&mut [f64; TANGENT]> = match tangent_ {
            Some(t) => Some(&mut t[i]),
            None => None
        };
        
        model(time, del_time, &del_strain_chunk, &mut stress_chunk, tangent_chunk, &mut history_chunk, &parameters);
    }
}