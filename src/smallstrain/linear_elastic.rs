use crate::smallstrain::SmallStrainModel;
use nalgebra::{DVectorView, DVectorViewMut, SMatrix, SVector};
use std::collections::HashMap;


pub struct LinearElastic3D {
    pub D: SMatrix<f64, 6, 6>,
}

/// implement the small strain model for linear elastic material
impl SmallStrainModel for LinearElastic3D {

    fn evaluate_ip(
        &self,
        ip: usize,
        del_t: f64,
        stress: &mut DVectorViewMut<f64>,
        del_strain: &DVectorView<f64>,
        tangent: &mut DVectorViewMut<f64>,
    ) {
        let mut view_stress = stress.fixed_view_mut::<6, 1>(ip * 6, 0);
        let mut view_tangent = tangent.fixed_view_mut::<36, 1>(ip * 36, 0);
        let view_strain = del_strain.fixed_view::<6, 1>(ip * 6, 0);
        view_stress += self.D * view_strain;
        view_tangent.copy_from_slice(&self.D.as_slice());
    }
}
impl LinearElastic3D {
    pub fn new(parameters: &HashMap<String, f64>) -> Self{
        //let mut D = SMatrix::<f64, 6, 6>::zeros();
        let E = parameters.get("E").unwrap();
        let nu = parameters.get("nu").unwrap();
        let mu = E / (2.0 * (1.0 + nu));
        let lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let D = SMatrix::<f64, 6, 6>::new(
            lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0,
            lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0,
            lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0,
            0.0,0.0,0.0,0.0,0.0, 2.0 * mu,
        );
        Self { D: D }
    }
}