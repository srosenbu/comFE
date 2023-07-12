use crate::smallstrain::SmallStrainModel;
use crate::interfaces::{ConstitutiveModel, QValues};
use nalgebra::{DVectorView, DVectorViewMut, SMatrix, SVector};
use std::collections::HashMap;


pub struct LinearElastic3D {
    pub D: SMatrix<f64, 6, 6>,
}

/// implement the small strain model for linear elastic material
// impl SmallStrainModel for LinearElastic3D {

//     fn evaluate_ip(
//         &self,
//         ip: usize,
//         del_t: f64,
//         stress: &mut DVectorViewMut<f64>,
//         del_strain: &DVectorView<f64>,
//         tangent: &mut DVectorViewMut<f64>,
//     ) {
//         let mut view_stress = stress.fixed_view_mut::<6, 1>(ip * 6, 0);
//         let mut view_tangent = tangent.fixed_view_mut::<36, 1>(ip * 36, 0);
//         let view_strain = del_strain.fixed_view::<6, 1>(ip * 6, 0);
//         view_stress += self.D * view_strain;
//         view_tangent.copy_from_slice(&self.D.as_slice());
//     }
// }
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

impl ConstitutiveModel for LinearElastic3D{
    fn evaluate_ip(
            &self,
            ip: usize,
            del_t: f64,
            input: &QValues<DVectorView<f64>>,
            output: &mut QValues<DVectorViewMut<f64>>,
        ) {
            let stress_0 = input.mandel_stress.unwrap().fixed_view::<6, 1>(ip * 6, 0);
            let mut view_tangent = output.mandel_tangent.unwrap().fixed_view_mut::<36, 1>(ip * 36, 0);
            let strain = input.mandel_strain.unwrap().fixed_view::<6, 1>(ip * 6, 0);
            let mut view_stress = output.mandel_stress.unwrap().fixed_view_mut::<6, 1>(ip * 6, 0);
            view_stress += self.D * strain;
            view_tangent.copy_from_slice(&self.D.as_slice());
    }
    fn define_input(&self) -> QValues<usize> {
       QValues::<usize> {
            mandel_strain: Some(&6),
            ..Default::default()
        }

    }
    fn define_output(&self) -> QValues<usize> {
       QValues::<usize> {
            mandel_stress: Some(&6),
            mandel_tangent: Some(&36),
            ..Default::default()
        }
    }
    fn n_quadrature_points(&self, input: &QValues<DVectorView<f64>>, output: &mut QValues<DVectorViewMut<f64>>)-> Result<usize, &str> {
        //TODO: check if the number of quadrature points in input and output are the same
        let n_ip = input.mandel_strain.unwrap().len() / 6;
        let n_ip_out = output.mandel_strain.unwrap().len() / 6;
        if n_ip != n_ip_out {
            return Err("The number of quadrature points in input and output are not the same");
        }
        Ok(n_ip)
    }
}