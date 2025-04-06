use nalgebra::{DVectorView, DVectorViewMut, SMatrix, SVector};
use std::convert::TryInto;

pub mod linear_elastic;

// define a type for a specific function signature
pub type SmallStrainModelFn<
    const STRESS_STRAIN_DIM: usize,
    const HISTORY_DIM: usize,
    const PARAMETER_DIM: usize,
> = fn(
    del_t: f64,
    stress: [f64; STRESS_STRAIN_DIM],
    del_strain: [f64; STRESS_STRAIN_DIM],
    parameters: [f64; PARAMETER_DIM],
    history: [f64; HISTORY_DIM],
) -> (
    [f64; STRESS_STRAIN_DIM],
    [[f64; STRESS_STRAIN_DIM]; STRESS_STRAIN_DIM],
    [f64; HISTORY_DIM],
);

pub fn elasticity_3d(
    del_t: f64,
    stress: [f64; 6],
    del_strain: [f64; 6],
    parameters: [f64; 2],
    history: [f64; 0],
) -> ([f64; 6], [[f64; 6]; 6], [f64; 0]) {
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
    let stress_vector = SVector::<f64, 6>::from(stress);
    let del_strain_vector = SVector::<f64, 6>::from(del_strain);
    let new_stress = stress_vector + D * del_strain_vector;
    let D_out = D.data.0;
    let new_stress = new_stress.data.0[0];
    let history = history;
    (new_stress, D_out, history)
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
    model: &SmallStrainModelFn<STRESS_STRAIN_DIM, HISTORY_DIM, PARAMETER_DIM>,
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

    let stress_chunks = stress.chunks_exact_mut(STRESS_STRAIN_DIM);
    let del_strain_chunks = del_strain.chunks_exact(STRESS_STRAIN_DIM);
    let history_chunks = history.chunks_exact_mut(HISTORY_DIM);
    let tangent_chunks = tangent.chunks_exact_mut(STRESS_STRAIN_DIM.pow(2));
    
    assert!(
            stress_chunks.len() == del_strain_chunks.len()
            && stress_chunks.len() == history_chunks.len()
            && stress_chunks.len() == tangent_chunks.len()
    );
    for (((stress_chunk, del_strain_chunk), history_chunk), tangent_chunk) in stress_chunks
        .zip(del_strain_chunks)
        .zip(history_chunks)
        .zip(tangent_chunks)
    {
        let stress: [f64; STRESS_STRAIN_DIM] = stress_chunk
            .try_into()
            .expect("Slice length does not match array length");
        let del_strain: [f64; STRESS_STRAIN_DIM] = del_strain_chunk
            .try_into()
            .expect("Slice length does not match array length");
        let history: [f64; HISTORY_DIM] = history_chunk
            .try_into()
            .expect("Slice length does not match array length");
        let (new_stress, D, new_history) = model(del_t, stress, del_strain, parameters, history);
        stress_chunk.copy_from_slice(&new_stress);
        history_chunk.copy_from_slice(&new_history);
        tangent_chunk.copy_from_slice(unsafe{std::slice::from_raw_parts(D.as_ptr() as *const f64, STRESS_STRAIN_DIM.pow(2))});
    }
    
}
