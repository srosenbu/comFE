use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    mandel_decomposition, mandel_rate_from_velocity_gradient, volumetric, MANDEL_IDENTITY,
};

use std::collections::HashMap;
#[derive(Debug)]
pub struct Hypoelasticity3D {
    mu: f64,
    lambda: f64,
}

impl ConstitutiveModel for Hypoelasticity3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            mu: *parameters.get("mu").unwrap(),
            lambda: *parameters.get("lambda").unwrap(),
        })
    }
    fn evaluate_ip(&self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let velocity_gradient = input
            .get_tensor::<{ Q::VelocityGradient.dim() }, { Q::VelocityGradient.size() }>(
                Q::VelocityGradient,
                ip,
            );
        let d_eps = mandel_rate_from_velocity_gradient(&velocity_gradient);
        let trace_d_eps = volumetric(&d_eps) * 3.0;
        let sigma_0 = input.get_vector::<{ Q::MandelStress.size() }>(Q::MandelStress, ip);

        let sigma_1 = sigma_0
            + 2. * self.mu * d_eps * del_t
            + self.lambda * trace_d_eps * del_t * MANDEL_IDENTITY;
        
        output.set_vector(Q::MandelStress, ip, sigma_1);
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::VelocityGradient, QDim::SquareTensor(3)),
        ])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions. These Variables are
    /// stored both in in the input and the output.
    fn define_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
        ])
    }

    /// Returns the physical quantities that are needed as output, but are not
    /// necessarily needed in oredr to calculate the constitutive model. An example is
    /// the consistent tangent which is not needed for the calculation of the stresses
    /// and is therefore purely an output quantity.
    fn define_output(&self) -> HashMap<Q, QDim> {
        HashMap::from([(Q::MandelStress, QDim::Vector(6))])
    }

    /// Returns the physical quantities that are optional output of the constitutive
    /// model. These quantities are not needed for the calculation of the stresses
    /// but can be useful for postprocessing.
    fn define_optional_output(&self) -> HashMap<Q, QDim> {
        HashMap::from([])
    }
    fn define_optional_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([])
    }
    
    fn parameters(&self)->HashMap<String, f64> {
        HashMap::from([
            ("mu".to_string(), self.mu),
            ("lambda".to_string(), self.lambda),
        ])
    }
}
