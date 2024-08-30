
use nalgebra::{SVector, SMatrix};
use crate::interfaces::*;
use crate::consts::*;
use core::ffi::c_double;
use std::collections::HashMap;


#[repr(C)]
struct LinearElasticity3D();

impl ConstitutiveModel<6, 36, 0, 2> for LinearElasticity3D {
    #[inline]
    fn evaluate(
        time: f64,
        del_time: f64,
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [f64; 36]>,
        history: &mut [f64; 0],
        parameters: &[f64; 2],
    ) {
        let mu = parameters[0];
        let lambda = parameters[1];
        let del_strain_vec = SVector::<f64, 6>::from_column_slice(del_strain);
        let stress_vec = SVector::<f64, 6>::from_column_slice(stress);
        let tangent_mat = SYM_ID_6_OUTER_SYM_ID_6 * lambda + (2.0 * mu) * ID_6;
        let new_stress =  stress_vec + tangent_mat * del_strain_vec;
        stress.copy_from_slice(new_stress.as_slice());
        if let Some(tangent) = tangent {
            tangent.copy_from_slice(tangent_mat.as_slice());
        }
    }
    fn parameters() -> std::collections::HashMap<String, (usize,usize)> {
        let mut map = HashMap::new();
        map.insert("mu".to_string(), (0,1));
        map.insert("lambda".to_string(), (1,1));
        map
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
    let del_strain=unsafe { &*(del_strain as *const [f64;6]) };
    let stress=unsafe { &mut *(stress as *mut [f64;6]) };
    let tangent=Some(unsafe { &mut *(tangent as *mut [f64;36]) });
    let history=unsafe { &mut *(history as *mut [f64;0]) };
    let parameters=unsafe { &*(parameters as *const [f64;2]) };
    LinearElasticity3D::evaluate(time, del_time, del_strain, stress, tangent, history, parameters);
}