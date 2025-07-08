use crate::consts::*;
use crate::{impl_array_equivalent, create_struct_with_field_names, q_dim_data_type};
use crate::interfaces::*;
use crate::mandel::*;
use core::ffi::c_double;
use nalgebra::{SMatrix, SVector, SVectorView, SVectorViewMut};
use phf::{phf_map, Map};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

const _: () = assert!(check_constitutive_model_maps::<6,36,0,0,2,2,LinearElasticity3D>());

const PARAMETERS_MAP_: [(&'static str, QDim); 2] = [("mu", QDim::Scalar), ("lambda", QDim::Scalar)];

#[repr(C)]
struct LinearElasticity3D();


create_struct_with_field_names!(
    LinearElasticityParameters,
    2,
    [(mu, (QDim::Scalar)),
    (kappa, (QDim::Scalar))]
);

create_struct_with_field_names!(
    LinearElasticityHistory,
    0,
    []
);
impl_array_equivalent!(LinearElasticityParameters, 2);
impl_array_equivalent!(LinearElasticityHistory, 0);

impl ConstitutiveModelFn<6, 0, 0, 2, 2> for LinearElasticity3D {

    const PARAMETERS_MAP: [(&'static str, QDim); 2] = LinearElasticityParameters::FIELDS;
    const HISTORY_MAP: [(&'static str, QDim); 0] = LinearElasticityHistory::FIELDS;

    type History = LinearElasticityHistory;
    type Parameters = LinearElasticityParameters;
    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        _strain: &[f64; 6],
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64;6]; 6]>,
        _history: &mut [f64; 0],
        parameters: &[f64; 2],
    ) {
        // Unpack parameters
        let parameters_ = LinearElasticityParameters::from_array(parameters);
        let mu = parameters_.mu;
        let kappa = parameters_.kappa;
        // Compute lambda from kappa and mu
        let lambda = kappa - (2.0 / 3.0) * mu;
        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);

        stress_vec += (del_strain_vec.trace() * lambda) * SYM_ID_6 + (2.0 * mu) * del_strain_vec;

        if let Some(tangent) = tangent {
            let tangent_mat = SYM_ID_6_OUTER_SYM_ID_6 * lambda + (2.0 * mu) * ID_6;
            *tangent = tangent_mat.data.0;
        }
    }
}


pub unsafe fn linear_elasticity3d_fn(
    time: c_double,
    del_time: c_double,
    strain: *const c_double,
    del_strain: *const c_double,
    stress: *mut c_double,
    tangent: *const c_double,
    history: *mut c_double,
    parameters: *const c_double,
) {
    let strain = unsafe { &*(strain as *const [f64; 6]) };
    let del_strain = unsafe { &*(del_strain as *const [f64; 6]) };
    let stress = unsafe { &mut *(stress as *mut [f64; 6]) };
    let tangent = Some(unsafe { &mut *(tangent as *mut [[f64; 6]; 6]) });
    let history = unsafe { &mut *(history as *mut [f64; 0]) };
    let parameters = unsafe { &*(parameters as *const [f64; 2]) };
    LinearElasticity3D::evaluate(
        time, del_time, strain, del_strain, stress, tangent, history, parameters,
    );
}
