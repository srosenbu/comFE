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
    fn check_dimensions() -> bool {
        let check = STRESS_STRAIN.pow(2) == TANGENT;
        let history_dim = Self::history().iter().fold(0, |acc, (name, dim, size)| {
            acc + size
        });
        let check = check && history_dim == HISTORY;
        check
    }

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
