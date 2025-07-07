use crate::consts::*;
use crate::interfaces::*;
use crate::mandel::*;
//use crate::impl_array_equivalent;
use crate::impl_from_array;
use core::ffi::c_double;
use nalgebra::{SMatrix, SVector, SVectorView, SVectorViewMut};
use phf::{Map, phf_map};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

const _: () = assert!(check_constitutive_model_maps::<
    6,
    36,
    2,
    7,
    4,
    4,
    MisesPlasticity3D,
>());

#[repr(C)]
struct MisesPlasticity3D();

#[repr(C)]
struct MisesPlasticityParameters {
    mu: f64,
    kappa: f64,
    y_0: f64,
    h: f64,
}

#[repr(C)]
struct MisesPlasticityHistory {
    alpha: f64,
    plastic_strain: [f64; 6],
}

//impl_array_equivalent!(MisesPlasticityParameters, 4);
//impl_array_equivalent!(MisesPlasticityHistory, 7);
impl_from_array!(MisesPlasticityParameters, 4);
impl_from_array!(MisesPlasticityHistory, 7);
//let stuff = MisesPlasticityParameters::from(&[1.,2.,3.,4.]);

impl ConstitutiveModelFn<6, 2, 7, 4, 4> for MisesPlasticity3D {
    const PARAMETERS_MAP: [(&'static str, Dim); 4] = [
        ("mu", Dim::Scalar),
        ("kappa", Dim::Scalar),
        ("y_0", Dim::Scalar),
        ("H", Dim::Scalar),
    ];
    const HISTORY_MAP: [(&'static str, Dim); 2] = [
        ("equivalent_plastic_strain", Dim::Scalar),
        ("plastic_strain", Dim::RotatableVector(6)),
    ];

    type History = MisesPlasticityHistory;
    type Parameters = MisesPlasticityParameters;

    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; 7],
        parameters: &[f64; 4],
    ) {
        let parameters_: &MisesPlasticityParameters = parameters.as_ref();
        let mu = parameters_.mu;
        let kappa = parameters_.kappa;
        let y_0 = parameters_.y_0;
        let h = parameters_.h;

        let a: [f64;4] = [1.0, 0.0, 0.0, 0.0];
        let b : &[f64;4] = a.as_ref();
        // Unpack history
        let history_: &MisesPlasticityHistory= history.as_ref();
        let alpha = history_.alpha;
        let mut plastic_strain_vec =
            SVectorView::<f64, 6>::from_array(&history_.plastic_strain);

        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);

        let (p_0, s_0) = stress_vec.vol_dev();
        let (eps_trace, eps_dev) = del_strain_vec.trace_dev();
        let p_1 = p_0 + kappa * eps_trace;

        let s_tr = s_0 + 2. * mu * eps_dev;
        let s_tr_eq = s_tr.mises_norm();

        let sigma_y = y_0 + h * alpha;

        //the .max(0.0) contains the check if the stress is already above the yield surface
        if s_tr_eq < sigma_y {
            // Elastic step
            stress_vec.copy_from(&(p_1 * SYM_ID_6 + s_tr));
            if let Some(tangent) = tangent {
                *tangent = (kappa * SYM_ID_6_OUTER_SYM_ID_6 + 2. * mu * PROJECTION_DEV_6)
                    .data
                    .0;
            }
            return;
        } else {
            let del_alpha = (s_tr_eq - sigma_y) / (3. * mu + h);
            let del_gamma = f64::sqrt(3. / 2.) * del_alpha;
            let theta = 1. - (3. * mu * del_alpha) / s_tr_eq;

            // Update the equivalent plastic strain
            // determine the plastic strain
            let n = s_tr / s_tr_eq;
            plastic_strain_vec += del_gamma * n;
            history_.alpha += del_alpha;

            stress_vec.copy_from(&(p_1 * SYM_ID_6 + theta * s_tr));

            if let Some(tangent) = tangent {
                let theta_bar = 1.0 / (1.0 + (h / (3.0 * mu))) - (1.0 - theta);
                let tangent_new = kappa * SYM_ID_6_OUTER_SYM_ID_6
                    + 2.0 * mu * theta * PROJECTION_DEV_6
                    + 2.0 * mu * theta_bar * n * n.transpose();
                // Copy the tangent matrix to the output
                *tangent = tangent_new.data.0;
            }
        }
    }
}
