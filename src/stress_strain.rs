use nalgebra::{DVectorView, DVectorViewMut, SMatrix, SVector, vector};

pub fn tensor_to_mandel(tensor: SMatrix<f64, 3, 3>) -> SVector<f64, 6> {
    //TODO: This is a hack. Replace once https://github.com/rust-lang/rust/issues/57241 is solved
    const SQRT: f64 = 1.4142135623730951;
    SVector::<f64, 6>::new(
        tensor.m11,
        tensor.m22,
        tensor.m33,
        SQRT * tensor.m23,
        SQRT * tensor.m13,
        SQRT * tensor.m12,
    )
}

pub fn mandel_to_tensor(mandel: SVector<f64, 6>) -> SMatrix<f64, 3, 3> {
    //TODO: This is a hack. Replace once https://github.com/rust-lang/rust/issues/57241 is solved
    const FACTOR: f64 = 0.7071067811865475;
    SMatrix::<f64, 3, 3>::new(
        mandel.x,
        FACTOR * mandel.b,
        FACTOR * mandel.a,
        FACTOR * mandel.b,
        mandel.y,
        FACTOR * mandel.w,
        FACTOR * mandel.a,
        FACTOR * mandel.w,
        mandel.z,
    )
}

pub fn volumetric(mandel: &SVector<f64, 6>) -> f64 {
    //TODO: I don't really like the notation with the x,y,z for the components,
    // but it is probably the fastest
    (mandel.x + mandel.y + mandel.z) / 3.0
}

pub fn add_volumetric(mandel: &mut SVector<f64, 6>, p: f64) {
    mandel.x += p;
    mandel.y += p;
    mandel.z += p;
}

pub fn deviatoric(mandel: &SVector<f64, 6>) -> SVector<f64, 6> {
    let p = volumetric(mandel);
    let mut dev = mandel.clone();
    add_volumetric(&mut dev, -p);
    dev
}

pub fn deviatoric_in_place(mandel: &mut SVector<f64, 6>) {
    let p = volumetric(mandel);
    add_volumetric(mandel, -p)
}

pub fn mandel_decomposition(mandel: &SVector<f64, 6>) -> (f64, SVector<f64, 6>) {
    let p = volumetric(mandel);
    let mut dev = mandel.clone();
    add_volumetric(&mut dev, -p);
    (-p, dev)
}

pub fn mandel_rate_from_velocity_gradient(
    velocity_gradient: &SMatrix<f64, 3, 3>,
) -> SVector<f64, 6> {
    // creates the strain rate mandel vector directly from the velocity gradient L instead of the rate
    // of deformation tensor D. Therefore, the factor is 1/sqrt(2) instead of sqrt(2)
    const FACTOR: f64 = 0.7071067811865475; // 1/sqrt(2)
    SVector::<f64, 6>::new(
        velocity_gradient.m11,
        velocity_gradient.m22,
        velocity_gradient.m33,
        FACTOR * (velocity_gradient.m23 + velocity_gradient.m32),
        FACTOR * (velocity_gradient.m13 + velocity_gradient.m31),
        FACTOR * (velocity_gradient.m12 + velocity_gradient.m21),
    )
}

pub const MANDEL_IDENTITY: SVector<f64, 6> = SVector::<f64, 6>::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

// fn jaumann_rotation_del_sigma(
//     velocity_gradient: &SMatrix<f64, 3, 3>,
//     stress: &SVector<f64, 6>,
// ) -> SVector<f64, 6> {
//     const FACTOR: f64 = 0.7071067811865475; // 1/sqrt(2)
//     let differences: [f64; 3] = [
//         velocity_gradient.m12 - velocity_gradient.m21,
//         velocity_gradient.m13 - velocity_gradient.m31,
//         velocity_gradient.m23 - velocity_gradient.m32,
//     ];
//     let mut result = SVector::<f64, 6>::zeros();
//     result.x = FACTOR * (stress.a * differences[1] + stress.b * differences[0]);
//     result.y = FACTOR * (stress.w * differences[2] - stress.b * differences[0]);
//     result.z = FACTOR * (-stress.w * differences[2] - stress.a * differences[1]);
//     result.w = FACTOR
//         * (-stress.y * differences[2] + stress.z * differences[2]
//             - FACTOR * (stress.a * differences[0] + stress.b * differences[1]));
//     result.a = FACTOR
//         * (-stress.x * differences[1] - stress.z * differences[1]
//             + FACTOR * (stress.w * differences[0] - stress.b * differences[2]));
//     result.b = FACTOR
//         * (-stress.x * differences[0]
//             + stress.y * differences[0]
//             + FACTOR * (stress.w * differences[1] + stress.a * differences[2]));
//     result
// }

pub fn jaumann_rotation(
    del_t: f64,
    velocity_gradient: &DVectorView<f64>,
    stress: &mut DVectorViewMut<f64>,
) {
    const FACTOR: f64 = 0.7071067811865475; // 1/sqrt(2)
    let factor_del_t = FACTOR * del_t;
    let n = velocity_gradient.len() / 9;
    let m = stress.len() / 6;
    assert!(
        n == m,
        "Velocity gradient and stress must have the same number of elements"
    );
    for i in 0..n {
        //let vel_grad = SMatrix::<f64, 3, 3>::from_row_slice(
        //    velocity_gradient.fixed_view::<9, 1>(i * 9, 0).as_slice(),
        //);
        let vel_grad = velocity_gradient.fixed_view::<9,1>(i * 9, 0);

        let mut stress_view = stress.fixed_view_mut::<6, 1>(i * 6, 0);
        //let v1 = vector![vel_grad.m12, vel_grad.m13, vel_grad.m23];
        //let v2 = vector![vel_grad.m21, vel_grad.m31, vel_grad.m32];

        // unsafe is okay because we index the fixed view of length 9. Bounds have been checked by fixed_view
        //let v1 = unsafe{vector![*vel_grad.get_unchecked(1), *vel_grad.get_unchecked(2), *vel_grad.get_unchecked(5)]};
        //let v2 = unsafe{vector![*vel_grad.get_unchecked(3), *vel_grad.get_unchecked(6), *vel_grad.get_unchecked(7)]};
        //let v2 = vector![vel_grad_row_slice[3], vel_grad_row_slice[6], vel_grad_row_slice[7]];
        //let diff = 0.5 * (v1 - v2);
        let diff:[f64; 3] = unsafe{[*vel_grad.get_unchecked(1) - *vel_grad.get_unchecked(3), *vel_grad.get_unchecked(2) - *vel_grad.get_unchecked(6), *vel_grad.get_unchecked(5) - *vel_grad.get_unchecked(7)]};
        
        //let diff = vel_grad - vel_grad.transpose();
        //let differences: [f64; 3] = [diff.m12, diff.m13, diff.m23];
        let stress_i = SVector::<f64, 6>::from_column_slice(stress_view.as_slice());
        stress_view.x += factor_del_t * (stress_i.a * diff[1] + stress_i.b * diff[0]);
        stress_view.y += factor_del_t * (stress_i.w * diff[2] - stress_i.b * diff[0]);
        stress_view.z += factor_del_t * (-stress_i.w * diff[2] - stress_i.a * diff[1]);
        stress_view.w += factor_del_t * (stress_i.z * diff[2] -stress_i.y * diff[2] - FACTOR * (stress_i.a * diff[0] + stress_i.b * diff[1]));
        stress_view.a += factor_del_t * (stress_i.z * diff[1] -stress_i.x * diff[1] + FACTOR * (stress_i.w * diff[0] - stress_i.b * diff[2]));
        stress_view.b += factor_del_t * (stress_i.y * diff[0] -stress_i.x * diff[0] + FACTOR * (stress_i.w * diff[1] + stress_i.a * diff[2]));
    }
}

pub fn jaumann_rotation_expensive(
    del_t: f64,
    velocity_gradient: &DVectorView<f64>,
    stress: &mut DVectorViewMut<f64>,
) {
    //const FACTOR:f64 = 0.7071067811865475; // 1/sqrt(2)
    let n = velocity_gradient.len() / 9;
    let m = stress.len() / 6;
    assert!(
        n == m,
        "Velocity gradient and stress must have the same number of elements"
    );
    for i in 0..n {
        let vel_grad = SMatrix::<f64, 3, 3>::from_row_slice(
            velocity_gradient.fixed_view::<9, 1>(i * 9, 0).as_slice(),
        );
        let mut stress_view = stress.fixed_view_mut::<6, 1>(i * 6, 0);
        let mut stress_tensor = mandel_to_tensor(stress_view.clone_owned());

        let W = 0.5 * (vel_grad - vel_grad.transpose());
        let W_sigma = W * stress_tensor;
        stress_tensor += del_t * (W_sigma + W_sigma.transpose());

        stress_view.copy_from(&tensor_to_mandel(stress_tensor));
    }
}
