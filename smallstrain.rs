use nalgebra::{DVectorView, DVectorViewMut, SMatrix, SVector, SVectorView, SVectorViewMut};
use std::convert::TryInto;

pub mod linear_elastic;

// define a type for a specific function signature
// pub type SmallStrainModelFn<
//     const STRESS_STRAIN_DIM: usize,
//     const HISTORY_DIM: usize,
//     const PARAMETER_DIM: usize,
// > = fn(
//     del_t: f64,
//     stress: &[f64; STRESS_STRAIN_DIM],
//     del_strain: &[f64; STRESS_STRAIN_DIM],
//     parameters: &[f64; PARAMETER_DIM],
//     history: &[f64; HISTORY_DIM],
// ) -> (
//     [f64; STRESS_STRAIN_DIM],
//     [[f64; STRESS_STRAIN_DIM]; STRESS_STRAIN_DIM],
//     [f64; HISTORY_DIM],
// );
pub type SmallStrainModelFn<
    const STRESS_STRAIN_DIM: usize,
    const HISTORY_DIM: usize,
    const PARAMETER_DIM: usize,
> = fn(
    del_t: f64,
    stress: &mut[f64; STRESS_STRAIN_DIM],
    del_strain: &[f64; STRESS_STRAIN_DIM],
    tangent: &mut[[f64; STRESS_STRAIN_DIM]; STRESS_STRAIN_DIM],
    parameters: &[f64; PARAMETER_DIM],
    history: &mut[f64; HISTORY_DIM],
);// -> (
//    [f64; STRESS_STRAIN_DIM],
//    [[f64; STRESS_STRAIN_DIM]; STRESS_STRAIN_DIM],
//    [f64; HISTORY_DIM],
//);

pub fn elasticity_3d(
    del_t: f64,
    stress: &mut[f64; 6],
    del_strain: &[f64; 6],
    tangent: &mut[[f64; 6]; 6],
    parameters: &[f64; 2],
    history: &mut[f64; 1],
) {//-> ([f64; 6], [[f64; 6]; 6], [f64; 1]) {
    let E = parameters[0];
    let nu = parameters[1];
    let mu = E / (2.0 * (1.0 + nu));
    let lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let c1 = lambda + 2.0 * mu;
    let c2 = 2. * mu;
    let D = SMatrix::<f64, 6, 6>::from([
        [c1, lambda, lambda, 0.0, 0.0, 0.0],
        [lambda, c1, lambda, 0.0, 0.0, 0.0],
        [lambda, lambda, c1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c2],
    ]);
    //let D = SMatrix::zeros();
    let mut stress_vector = unsafe{SVectorViewMut::<f64, 6>::from_slice_unchecked(stress, 0)};
    let del_strain_vector = unsafe{SVectorView::<f64, 6>::from_slice_unchecked(del_strain,0)};
    stress_vector += D * del_strain_vector;
    //let D_out = D.data.0;
    tangent.copy_from_slice(&D.data.0);
    //let new_stress = stress_vector + D * del_strain_vector;
    //let D_out = D.data.0;
    //let new_stress = new_stress.data.0[0];
    //let history = history;
    //(new_stress.data.0[0], D.data.0, *history)
}
pub struct SmallStrainModel<
    const STRESS_STRAIN_DIM: usize,
    const HISTORY_DIM: usize,
    const PARAMETER_DIM: usize,
> {
    pub model: SmallStrainModelFn<STRESS_STRAIN_DIM, HISTORY_DIM, PARAMETER_DIM>,
    pub parameters: [f64; PARAMETER_DIM],
}

pub fn evaluate_model<
    const STRESS_STRAIN_DIM: usize,
    const HISTORY_DIM: usize,
    const PARAMETER_DIM: usize,
>(
    model: SmallStrainModelFn<STRESS_STRAIN_DIM, HISTORY_DIM, PARAMETER_DIM>,
    del_t: f64,
    stress: &mut [f64],
    del_strain: &[f64],
    parameters: &[f64],
    history: &mut [f64],
    tangent: &mut [f64],
) {
    let parameters: [f64; PARAMETER_DIM] = parameters
        .try_into()
        .expect("Slice length does not match array length");

    //let stress_chunks = stress.chunks_exact_mut(STRESS_STRAIN_DIM);
    //let del_strain_chunks = del_strain.chunks_exact(STRESS_STRAIN_DIM);
    //let history_chunks = history.chunks_exact_mut(HISTORY_DIM);
    //let tangent_chunks = tangent.chunks_exact_mut(STRESS_STRAIN_DIM.pow(2));
    let stress_len = stress.len() / STRESS_STRAIN_DIM;
    let strain_len = del_strain.len() / STRESS_STRAIN_DIM;
    let history_len = history.len() / HISTORY_DIM;
    let tangent_len = tangent.len() / STRESS_STRAIN_DIM.pow(2);
    //assert!(
    //    stress_chunks.len() == del_strain_chunks.len()
    //        && stress_chunks.len() == history_chunks.len()
    //        && stress_chunks.len() == tangent_chunks.len()
    //);
    assert!(
        stress_len == strain_len
            && stress_len == history_len
            && stress_len == tangent_len
    );
    let stress_ = unsafe {
        std::slice::from_raw_parts_mut(
            stress.as_mut_ptr() as *mut [f64; STRESS_STRAIN_DIM],
            stress_len,
        )
    };
    let strain_ = unsafe {
        std::slice::from_raw_parts(
            del_strain.as_ptr() as *const [f64; STRESS_STRAIN_DIM],
            strain_len,
        )
    };
    let history_ = unsafe {
        std::slice::from_raw_parts_mut(
            history.as_mut_ptr() as *mut [f64; HISTORY_DIM],
            history_len,
        )
    };
    let tangent_ = unsafe {
        std::slice::from_raw_parts_mut(
            tangent.as_mut_ptr() as *mut [[f64; STRESS_STRAIN_DIM]; STRESS_STRAIN_DIM],
            tangent_len,
        )
    };
    //for (((stress_chunk, del_strain_chunk), history_chunk), tangent_chunk) in stress_.zip(strain_).zip(history_).zip(tangent_) {
        //.zip(del_strain_chunks)
        //.zip(history_chunks)
        //.zip(tangent_chunks)
    for i in 0..stress_len {
        let mut stress_chunk = stress_[i];
        let del_strain_chunk = strain_[i];
        let mut history_chunk = history_[i];
        let mut tangent_chunk = tangent_[i];
        // let stress: [f64; STRESS_STRAIN_DIM] = stress_chunk
        //     .try_into()
        //     .expect("Slice length does not match array length");
        // let del_strain: [f64; STRESS_STRAIN_DIM] = del_strain_chunk
        //     .try_into()
        //     .expect("Slice length does not match array length");
        // let history: [f64; HISTORY_DIM] = history_chunk
        //     .try_into()
        //     .expect("Slice length does not match array length");
        //(stress_[i], tangent_[i], history_[i]) = model(del_t, &stress_chunk, &del_strain_chunk, &parameters, &history_chunk);
        model(del_t, &mut stress_chunk, &del_strain_chunk, &mut tangent_chunk, &parameters, &mut history_chunk);
        //tangent_[i].copy_from_slice(unsafe {
        //    std::slice::from_raw_parts(D.as_ptr() as *const [[f64; STRESS_STRAIN_DIM]; STRESS_STRAIN_DIM], STRESS_STRAIN_DIM.pow(2))
        //});
        //stress_[i] = new_stress;
        //history_[i] = new_history;
        //tangent_[i] = D;
        //tangent_chunk.copy_from_slice(unsafe {
        //    std::slice::from_raw_parts(D.as_ptr() as *const f64, STRESS_STRAIN_DIM.pow(2))
        //});
    }
    // {
    //     let stress: [f64; STRESS_STRAIN_DIM] = stress_chunk
    //         .try_into()
    //         .expect("Slice length does not match array length");
    //     let del_strain: [f64; STRESS_STRAIN_DIM] = del_strain_chunk
    //         .try_into()
    //         .expect("Slice length does not match array length");
    //     let history: [f64; HISTORY_DIM] = history_chunk
    //         .try_into()
    //         .expect("Slice length does not match array length");
    //     let (new_stress, D, new_history) = model(del_t, stress, del_strain, parameters, history);
    //     stress_chunk.copy_from_slice(&new_stress);
    //     history_chunk.copy_from_slice(&new_history);
    //     tangent_chunk.copy_from_slice(unsafe {
    //         std::slice::from_raw_parts(D.as_ptr() as *const f64, STRESS_STRAIN_DIM.pow(2))
    //     });
    // }
}
