use crate::consts::*;
use crate::{impl_array_equivalent, create_struct_with_field_names};
use crate::interfaces::*;
use crate::mandel::*;
use core::ffi::c_double;
use nalgebra::{SMatrix, SVector, SVectorView, SVectorViewMut};
use phf::{phf_map, Map};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

const _: () = assert!(check_constitutive_model_maps::<6,36,0,0,2,2,LinearElasticity3D>());

const PARAMETERS_MAP_: [(&'static str, Dim); 2] = [("mu", Dim::Scalar), ("lambda", Dim::Scalar)];

#[repr(C)]
struct LinearElasticity3D();

create_struct_with_field_names!(
    LinearElasticityParameters,
    [(mu, f64),
    (lambda, f64)]
);
create_struct_with_field_names!(
    LinearElasticityHistory,
    []
);
impl_array_equivalent!(LinearElasticityParameters, 2);
impl_array_equivalent!(LinearElasticityHistory, 0);

impl ConstitutiveModelFn<6, 0, 0, 2, 2> for LinearElasticity3D {

    const PARAMETERS_MAP: [(&'static str, Dim); 2] = PARAMETERS_MAP_;
    const HISTORY_MAP: [(&'static str, Dim); 0] = [];

    type History = LinearElasticityHistory;
    type Parameters = LinearElasticityParameters;
    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64;6]; 6]>,
        _history: &mut [f64; 0],
        parameters: &[f64; 2],
    ) {
        // Unpack parameters
        let parameters_ = LinearElasticityParameters::from_array(parameters);
        let mu = parameters_.mu;
        let lambda = parameters_.lambda;
        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);

        stress_vec += (del_strain_vec.trace() * lambda) * SYM_ID_6 + (2.0 * mu) * del_strain_vec;

        if let Some(tangent) = tangent {
            let tangent_mat = SYM_ID_6_OUTER_SYM_ID_6 * lambda + (2.0 * mu) * ID_6;
            let tangent_flat = tangent.as_flattened_mut();
            tangent_flat.copy_from_slice(tangent_mat.as_slice());
        }
    }
}


pub unsafe fn linear_elasticity3d_fn(
    time: c_double,
    del_time: c_double,
    del_strain: *const c_double,
    stress: *mut c_double,
    tangent: *const c_double,
    history: *mut c_double,
    parameters: *const c_double,
) {
    let del_strain = unsafe { &*(del_strain as *const [f64; 6]) };
    let stress = unsafe { &mut *(stress as *mut [f64; 6]) };
    let tangent = Some(unsafe { &mut *(tangent as *mut [[f64; 6]; 6]) });
    let history = unsafe { &mut *(history as *mut [f64; 0]) };
    let parameters = unsafe { &*(parameters as *const [f64; 2]) };
    LinearElasticity3D::evaluate(
        time, del_time, del_strain, stress, tangent, history, parameters,
    );
}
