use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use nalgebra::SMatrix;
use std::collections::HashMap;

#[derive(Debug)]
pub struct LinearElastic3D {
    pub D: SMatrix<f64, 6, 6>,
    pub E: f64,
    pub nu: f64,
}



impl ConstitutiveModel for LinearElastic3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        let E = parameters.get("E").unwrap();
        let nu = parameters.get("nu").unwrap();
        let mu = E / (2.0 * (1.0 + nu));
        let lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let D = SMatrix::<f64, 6, 6>::new(
            lambda + 2.0 * mu,
            lambda,
            lambda,
            0.0,
            0.0,
            0.0,
            lambda,
            lambda + 2.0 * mu,
            lambda,
            0.0,
            0.0,
            0.0,
            lambda,
            lambda,
            lambda + 2.0 * mu,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * mu,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * mu,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * mu,
        );
        Some(Self { D: D , E: *E, nu: *nu})
    }
    fn evaluate_ip(&self, ip: usize, _del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let strain = input.get_vector::<{Q::MandelStrain.size()}>(Q::MandelStrain, ip);

        let new_stress = self.D * strain;
        
        output.set_vector(Q::MandelStress, ip, new_stress);
        output.set_slice::<36>(Q::MandelTangent, ip, &self.D.as_slice());
    }

    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([(Q::MandelStrain, Q::MandelStrain.q_dim())])
    }

    fn define_output(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, Q::MandelStress.q_dim()),
            (Q::MandelTangent, Q::MandelTangent.q_dim()),
        ])
    }
    fn parameters(&self)-> HashMap<String, f64>{
        HashMap::from([
            ("E".to_string(), self.E),
            ("nu".to_string(), self.nu),
        ])
    }
    
}
