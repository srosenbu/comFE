use std::collections::HashMap;
use std::num::NonZeroUsize;

use crate::consts::*;
use konst::{const_eq, eq_str};
use nalgebra::{SMatrix, SVector, SVectorView, Scalar};
use phf::{Map, OrderedMap};
use serde::Serialize;

#[repr(C)]
pub enum Dim {
    Scalar,
    Vector(usize),
    Matrix(usize),
    RotatableVector(usize),
    RotatableMatrix(usize),
}
impl Dim {
    pub const fn size(&self) -> usize {
        match self {
            Dim::Scalar => 1,
            Dim::Vector(size) => *size,
            Dim::RotatableVector(size) => *size,
            Dim::Matrix(size) => *size,
            Dim::RotatableMatrix(size) => *size,
        }
    }
    pub const fn len(&self) -> usize {
        match self {
            Dim::Scalar => 1,
            Dim::Vector(size) => *size,
            Dim::RotatableVector(size) => *size,
            Dim::Matrix(size) => *size * *size,
            Dim::RotatableMatrix(size) => *size * *size,
        }
    }
}
pub trait ArrayEquivalent<const N:usize> : Sized{
    fn from_array(array: &[f64; N]) -> &Self;
    fn from_array_mut(array: &mut [f64; N]) -> &mut Self;

    fn as_array(&self) -> &[f64; N];
    fn as_array_mut(&mut self) -> &mut [f64; N];

}
#[macro_export]
macro_rules! create_struct_with_field_names {
    ($struct_name:ident, [$(($field_name:ident, $type:ty)),*]) => {
        struct $struct_name {
            $(
                $field_name: $type,
            )*
        }

        impl $struct_name {
            pub const FIELD_NAMES: &'static [&'static str] = &[
                $(stringify!($field_name)),*
            ];
        }
    };
}


#[macro_export] macro_rules! impl_from_array {
    ($type:ty, $size:expr) => {
        const _: () = assert!(std::mem::size_of::<[f64; $size]>() == std::mem::size_of::<$type>(), "size mismatch");
        
        impl AsRef<[f64;$size]> for $type {
            fn as_ref(&self) -> &[f64;$size] {
                unsafe { &*(self as *const Self as *const [f64; $size]) }
            }
        }
        impl AsRef<$type> for [f64;$size] {
            fn as_ref(&self) -> &$type {
                unsafe { &*(self as *const [f64;$size] as *const $type )}
            }
        }
        impl From<&[f64;$size]> for &$type {
            fn from(array: &[f64;$size]) -> Self {
                unsafe { *(array as *const [f64; $size] as *const Self) }
            }
        }
    }
    //         #[inline]
    //         fn from_array(array: &[f64; $size]) -> &Self {
    //             unsafe { &*(array as *const [f64; $size] as *const Self) }
    //         }

    //         #[inline]
    //         fn from_array_mut(array: &mut [f64; $size]) -> &mut Self {
    //             unsafe { &mut *(array as *mut [f64; $size] as *mut Self) }
    //         }

    //         #[inline]
    //         fn as_array(&self) -> &[f64; $size] {
    //             unsafe { &*(self as *const Self as *const [f64; $size]) }
    //         }

    //         #[inline]
    //         fn as_array_mut(&mut self) -> &mut [f64; $size] {
    //             unsafe { &mut *(self as *mut Self as *mut [f64; $size]) }
    //         }
    //     }
    // };
}
pub trait ConstitutiveModelFn<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>
    where 
        [f64;PARAMETERS]: AsRef<Self::Parameters>,
        [f64;HISTORY]: AsRef<Self::History>,
{
    const HISTORY_MAP: [(&'static str, Dim); N_HISTORY];
    const PARAMETERS_MAP: [(&'static str, Dim); N_PARAMETERS];

    type History: AsRef<[f64;HISTORY]>;
    type Parameters: AsRef<[f64;PARAMETERS]>;

    fn evaluate(
        time: f64,
        del_time: f64,
        del_strain: &[f64; STRESS_STRAIN],
        stress: &mut [f64; STRESS_STRAIN],
        tangent: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]>,
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
    T: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>(
    name: &'static str,
) -> (usize, usize)
where
    T: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
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
    T: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>(
    name: &'static str,
) -> (usize, usize)
where
    T: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
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
    model: MODEL,
    time: f64,
    del_time: f64,
    del_strain: &[f64],
    stress: &mut [f64],
    tangent: Option<&mut [f64]>,
    history: &mut [f64],
    parameters: &[f64],
) where
    MODEL: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
{
    let parameters: [f64; PARAMETERS] = parameters
        .try_into()
        .expect("Slice length does not match array length");

    let (stress_, stress_rest) = stress.as_chunks_mut::<STRESS_STRAIN>();
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
    let history_len = history_.len();
    let tangent_len = tangent_.as_ref().map_or(0, |t| t.len());

    assert!(
        stress_len == del_strain_len
            && stress_len == history_len
            && (stress_len == tangent_len || tangent_.is_none()),
        "Stress, strain, history, and tangent lengths do not match: \
        stress_len: {}, del_strain_len: {}, history_len: {}, tangent_len: {}",
        stress_len,
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
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = del_strain_[i];
        let mut history_chunk = history_[i];
        let tangent_chunk: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]> =
            tangent_.as_mut().map(|t| &mut t[i]);
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
// pub fn evaluate_model_from_maps<
//     const STRESS_STRAIN: usize,
//     const TANGENT: usize,
//     const N_HISTORY: usize,
//     const HISTORY: usize,
//     const N_PARAMETERS: usize,
//     const PARAMETERS: usize,
//     MODEL,
// >(
//     model: MODEL,
//     time: f64,
//     del_time: f64,
//     del_strain: &[f64],
//     stress: &mut [f64],
//     tangent: Option<&mut [f64]>,
//     history: HashMap<&str,&mut [f64]>,
//     parameters: &[f64],
// ) where
//     MODEL: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
// {
//     let parameters: [f64; PARAMETERS] = parameters
//         .try_into()
//         .expect("Slice length does not match array length");

//     let (stress_, stress_rest) = stress.as_chunks_mut::<STRESS_STRAIN>();
//     let (del_strain_, del_strain_rest) = del_strain.as_chunks::<STRESS_STRAIN>();
//     //let (history_, history_rest) = history.as_chunks_mut::<HISTORY>();
//     let mut tangent_ = {
//         match tangent {
//             Some(t) => {
//                 let (tangent_, tangent_rest_1) = t.as_chunks_mut::<STRESS_STRAIN>();
//                 let (tangent_, tangent_rest_2) = tangent_.as_chunks_mut::<STRESS_STRAIN>();
//                 Some(tangent_)
//             }
//             None => None,
//         }
//     };

//     // extract history into an array of slices

//     let mut history_: [&mut [f64];N_HISTORY];
//     for (i,(name, dim)) in MODEL::HISTORY_MAP.iter().enumerate() {
//         //let (start, len) = get_history_index::<STRESS_STRAIN, TANGENT, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS, MODEL>(name);
//         history_[i] = *history.get(name).expect("History variable not found in model");
//     }

//     let stress_len = stress_.len();
//     let del_strain_len = del_strain_.len();
//     let history_len = history_.len();
//     let tangent_len = tangent_.as_ref().map_or(0, |t| t.len());

//     assert!(
//         stress_len == del_strain_len
//             && stress_len == history_len
//             && (stress_len == tangent_len || tangent_.is_none()),
//         "Stress, strain, history, and tangent lengths do not match: \
//         stress_len: {}, del_strain_len: {}, history_len: {}, tangent_len: {}",
//         stress_len,
//         del_strain_len,
//         history_len,
//         tangent_len
//     );
//     assert!(
//         stress_rest.is_empty()
//             && del_strain_rest.is_empty()
//             && history_rest.is_empty()
//             && tangent_.as_ref().map_or(true, |t| t.is_empty()),
//         "Input slices are not of the correct length: \
//         stress_rest: {:?}, del_strain_rest: {:?}, history_rest: {:?}",
//         stress_rest,
//         del_strain_rest,
//         history_rest
//     );

//     for i in 0..stress_len {
//         let mut stress_chunk = stress_[i];
//         let del_strain_chunk = del_strain_[i];
//         let mut history_chunk = history_[i];
//         let tangent_chunk: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]> =
//             tangent_.as_mut().map(|t| &mut t[i]);
//         MODEL::evaluate(
//             time,
//             del_time,
//             &del_strain_chunk,
//             &mut stress_chunk,
//             tangent_chunk,
//             &mut history_chunk,
//             &parameters,
//         );
//     }
// }
