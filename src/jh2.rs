use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    deviatoric, mandel_decomposition, mandel_rate_from_velocity_gradient, MANDEL_IDENTITY,
};

use nalgebra::SMatrix;
use std::cmp;
use std::collections::HashMap;

#[derive(Debug)]
struct JH2Parameters {
    RHO: f64,
    SHEAR_MODULUS: f64,
    A: f64,
    B: f64,
    C: f64,
    M: f64,
    N: f64,
    EPS0: f64,
    T: f64,
    SIGMAHEL: f64,
    PHEL: f64,
    D1: f64,
    D2: f64,
    K1: f64,
    K2: f64,
    K3: f64,
    BETA: f64,
    EFMIN: f64,
}
#[derive(Debug)]
pub struct JH23D {
    parameters: JH2Parameters,
}

impl ConstitutiveModel for JH23D {
    fn new(parameters: &HashMap<String, f64>) -> Self {
        Self {
            parameters: JH2Parameters {
                RHO: *parameters.get("RHO").unwrap(),
                SHEAR_MODULUS: *parameters.get("SHEAR_MODULUS").unwrap(),
                A: *parameters.get("A").unwrap(),
                B: *parameters.get("B").unwrap(),
                C: *parameters.get("C").unwrap(),
                M: *parameters.get("M").unwrap(),
                N: *parameters.get("N").unwrap(),
                EPS0: *parameters.get("EPS0").unwrap(),
                T: *parameters.get("T").unwrap(),
                SIGMAHEL: *parameters.get("SIGMAHEL").unwrap(),
                PHEL: *parameters.get("PHEL").unwrap(),
                D1: *parameters.get("D1").unwrap(),
                D2: *parameters.get("D2").unwrap(),
                K1: *parameters.get("K1").unwrap(),
                K2: *parameters.get("K2").unwrap(),
                K3: *parameters.get("K3").unwrap(),
                BETA: *parameters.get("BETA").unwrap(),
                EFMIN: *parameters.get("EFMIN").unwrap(),
            },
        }
    }
    fn evaluate_ip(&self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let velocity_gradient = input
            .get_tensor::<{ Q::VelocityGradient.dim() }, { Q::VelocityGradient.size() }>(
                Q::VelocityGradient,
                ip,
            );
        let d_eps = mandel_rate_from_velocity_gradient(&velocity_gradient);

        let sigma_0 = input.get_vector::<{ Q::MandelStress.size() }>(Q::MandelStress, ip);

        let mut del_lambda = 0.0;

        let damage_0 = input.get_scalar(Q::Damage, ip);
        let mut damage_1 = damage_0;

        let (p_0, s_0) = mandel_decomposition(&sigma_0);
        let s_tr = s_0 + 2. * self.parameters.SHEAR_MODULUS * (deviatoric(&d_eps)) * del_t;
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();
        let d_eps_eq = ((2. / 3.) * d_eps.norm_squared()).sqrt();
        let mut alpha;

        let p_s = p_0 / self.parameters.PHEL;
        let t_s = self.parameters.T / self.parameters.PHEL;
        let mut rate_factor = 1.;

        let fracture_surface =
            (self.parameters.A * (p_s + t_s).powf(self.parameters.N) * self.parameters.SIGMAHEL)
                .max(0.0);
        let residual_surface =
            (self.parameters.B * (p_s).powf(self.parameters.M) * self.parameters.SIGMAHEL).max(0.0);
        if d_eps_eq >= self.parameters.EPS0 {
            rate_factor += self.parameters.C * (d_eps_eq / self.parameters.EPS0).ln();
        }
        let yield_surface = rate_factor * {
            if damage_0 == 0.0 {
                fracture_surface
            } else {
                fracture_surface * (1. - damage_0) + damage_0 * residual_surface
            }
        };
        if s_tr_eq > yield_surface {
            let e_p_f = (self.parameters.D1 * (p_s + t_s).powf(self.parameters.D2))
                .max(self.parameters.EFMIN);

            del_lambda = (s_tr_eq - yield_surface) / (3. * self.parameters.SHEAR_MODULUS);
            alpha = yield_surface / s_tr_eq;

            damage_1 = (damage_0 + del_lambda / e_p_f).min(1.0);
            output.set_scalar(Q::Damage, ip, damage_1);
        } else {
            alpha = 1.0;
        }

        // /***********************************************************************
        //  * UPDATE DENSITY
        //  * The density is updated using the explicit midpoint rule for the
        //  * deformation gradient.
        //  TODO: Move this since, it will be calculated outside of the constitutive model
        //  **********************************************************************/
        let f1 = del_t / 2. * velocity_gradient.trace();
        let density_0 = input.get_scalar(Q::Density, ip);
        let density_1 = density_0 * (1. - f1) / (1. + f1);
        output.set_scalar(Q::Density, ip, density_1);

        let mu = density_1 / self.parameters.RHO - 1.;
        
        //let mut del_p_0 = 0.0;
        
        let p = {
            if mu > 0.0 {
                self.parameters.K1 * mu
                    + self.parameters.K2 * mu.powi(2)
                    + self.parameters.K3 * mu.powi(3)
                    + input.get_scalar(Q::BulkingPressure, ip)
            } else {
                (self.parameters.K1 * mu).max(-self.parameters.T * (1. - damage_1))
            }
        };
        if damage_1 > damage_0 {
            let y_old = damage_0 * residual_surface + (1. - damage_0) * fracture_surface;
            let y_new = damage_1 * residual_surface + (1. - damage_1) * fracture_surface;
            let u_old = y_old.powi(2) / (6. * self.parameters.SHEAR_MODULUS);
            let u_new = y_new.powi(2) / (6. * self.parameters.SHEAR_MODULUS);

            let del_u = u_old - u_new;

            let del_p_0 = input.get_scalar(Q::BulkingPressure, ip);
            let del_p = -self.parameters.K1 * mu
                + ((self.parameters.K1 * mu + del_p_0).powi(2)
                    + 2. * self.parameters.BETA * self.parameters.K1 * del_u)
                    .sqrt();
            output.set_scalar(Q::BulkingPressure, ip, del_p);
        }

        // /***********************************************************************
        //  * Combine deviatoric and volumetric stresses
        //  **********************************************************************/
        let sigma_1 = alpha * s_tr - MANDEL_IDENTITY * p;
        output.set_vector(Q::MandelStress, ip, sigma_1);

        // ***********************************************************************
        // Update optional output variables if needed
        // **********************************************************************

        if output.is_some(Q::EqPlasticStrain) {
            output.add_scalar(Q::EqPlasticStrain, ip, del_lambda);
        }
        if output.is_some(Q::EqStrainRate) {
            output.set_scalar(Q::EqStrainRate, ip, d_eps_eq);
        }
        if output.is_some(Q::MandelStrainRate) {
            output.set_vector(Q::MandelStrainRate, ip, d_eps);
        }
        if output.is_some(Q::MisesStress) {
            output.set_scalar(Q::MisesStress, ip, alpha * s_tr_eq);
        }
        if output.is_some(Q::Pressure) {
            output.set_scalar(Q::Pressure, ip, p);
        }
        if output.is_some(Q::InternalEnergyRate) {
            output.set_scalar(Q::InternalEnergyRate, ip, sigma_1.dot(&d_eps));
        }
        if output.is_some(Q::InternalElasticEnergyRate) {
            output.set_scalar(Q::InternalElasticEnergyRate, ip, sigma_1.dot());
        }
            
        
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::VelocityGradient, QDim::SquareTensor(3)),
            (Q::MandelStress, QDim::Vector(6)),
        ])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions.
    fn define_internal_variables(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::Damage, QDim::Scalar),
            (Q::BulkingPressure, QDim::Scalar),
            (Q::Density, QDim::Scalar),
        ])
    }

    /// Returns the physical quantities that are needed as output, but are not
    /// necessarily needed in oredr to calculate the constitutive model. An example is
    /// the consistent tangent which is not needed for the calculation of the stresses 
    /// and is therefore purely an output quantity.
    fn define_output(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
        ])
    }

    /// Returns the physical quantities that are optional output of the constitutive
    /// model. These quantities are not needed for the calculation of the stresses
    /// but can be useful for postprocessing.
    fn define_optional_output(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::EqStrainRate, QDim::Scalar),
            (Q::MandelStrainRate, QDim::Vector(6)),
            (Q::MisesStress, QDim::Scalar),
            (Q::Pressure, QDim::Scalar),
            (Q::InternalEnergyRate, QDim::Scalar),
            (Q::InternalElasticEnergyRate, QDim::Scalar),
        ])
    }

}
