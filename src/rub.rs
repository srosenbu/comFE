use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    mandel_decomposition, mandel_rate_from_velocity_gradient, MANDEL_IDENTITY,
};

use core::panic;
use std::collections::HashMap;

#[derive(Debug)]
pub struct RUBConstParameters {
    pub density: f64,
    pub shear_modulus: f64,
    pub bulk_modulus: f64,
    pub a_y: f64,
    pub b_y: f64,
    pub d_y: f64,
    pub a_r: f64,
    pub b_r: f64,
    pub e_f: f64,
    pub h: f64,
    pub alpha_0: f64,
}
#[derive(Debug)]
pub struct RUB3D {
    parameters: RUBConstParameters,
}

impl ConstitutiveModel for RUB3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            parameters: RUBConstParameters {
                density: *parameters.get("density").unwrap(),
                shear_modulus: *parameters.get("shear_modulus").unwrap(),
                bulk_modulus: *parameters.get("bulk_modulus").unwrap(),
                a_y: *parameters.get("a_y").unwrap(),
                b_y: *parameters.get("b_y").unwrap(),
                d_y: *parameters.get("d_y").unwrap(),
                a_r: *parameters.get("a_r").unwrap(),
                b_r: *parameters.get("b_r").unwrap(),
                e_f: *parameters.get("e_f").unwrap(),
                h: *parameters.get("h").unwrap(),
                alpha_0: *parameters.get("alpha_0").unwrap(),
            },
        })
    }
    fn evaluate_ip(&mut self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let velocity_gradient = input
            .get_tensor::<{ Q::VelocityGradient.dim() }, { Q::VelocityGradient.size() }>(
                Q::VelocityGradient,
                ip,
            );
        let d_eps = mandel_rate_from_velocity_gradient(&velocity_gradient);
        let (mut d_eps_vol, d_eps_dev) = mandel_decomposition(&d_eps);
        d_eps_vol *= -1.0;

        let sigma_0 = input.get_vector::<{ Q::MandelStress.size() }>(Q::MandelStress, ip);

        let lambda_0 = input.get_scalar(Q::EqPlasticStrain, ip);
        let mut del_lambda = 0.0;

        let damage_0 = input.get_scalar(Q::Damage, ip);
        let mut damage_1 = damage_0;

        let hardening_factor = 1.0+self.parameters.h*lambda_0;

        let (p_0, s_0) = mandel_decomposition(&sigma_0);
        let p_0 = p_0 - input.get_scalar(Q::BulkViscosity, ip);

        let s_tr = s_0 + 2. * self.parameters.shear_modulus * d_eps_dev * del_t;
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();
        let d_eps_eq = ((2. / 3.) * d_eps.norm_squared()).sqrt();
        let mut alpha: f64;

        let initial_yield_slope = self.parameters.b_y / self.parameters.a_y;
        let mut initial_yield_surface = f64::sqrt(3.0)
            * initial_yield_slope
            * f64::sqrt((-3.0 * p_0 - self.parameters.d_y).powi(2) - self.parameters.a_y.powi(2));

        let residual_slope = self.parameters.b_r / self.parameters.a_r;
        let mut residual_surface = f64::sqrt(3.0)
            * residual_slope
            * f64::sqrt((-3.0 * p_0 - self.parameters.a_r).powi(2) - self.parameters.a_r.powi(2));
        initial_yield_surface = {
            if initial_yield_surface.is_nan() {
                0.0
            } else {
                initial_yield_surface
            }
        };
        
        residual_surface = {
            if residual_surface.is_nan() {
                0.0
            } else {
                residual_surface
            }
        };
        
        let yield_surface =  {
            if damage_0 == 0.0 {
                initial_yield_surface * hardening_factor
            } else {
                initial_yield_surface * (1. - damage_0) * hardening_factor + damage_0 * residual_surface
            }
        };
        if s_tr_eq > yield_surface {
            del_lambda = (s_tr_eq - yield_surface) / (3. * self.parameters.shear_modulus);
            alpha = yield_surface / s_tr_eq;

            let lambda_new = lambda_0 + del_lambda;
            damage_1 = (1. - ((self.parameters.alpha_0 - lambda_new) / self.parameters.e_f).exp())
                .max(0.0);

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
        let f1 = del_t / 2. * 3. * d_eps_vol;
        let density_0 = input.get_scalar(Q::Density, ip);
        let density_1 = density_0 * (1. - f1) / (1. + f1);
        debug_assert!(
            density_1 > 0.0,
            "Negative density encountered in JH2 model: {}",
            density_1
        );
        output.set_scalar(Q::Density, ip, density_1);

        let mu = density_1 / self.parameters.density - 1.;
        let tensile_limit = -(self.parameters.d_y - self.parameters.a_y) / 3.0 ;
        let p_1 = (self.parameters.bulk_modulus * mu).max((1. - damage_1)*tensile_limit);

        // Calculate bulk viscosity
        let l = input.get_scalar(Q::CellDiameter, ip);
        let tr_deps = d_eps_vol * 3.;
        let c = (self.parameters.bulk_modulus / self.parameters.density).sqrt();
        let q_1 = {
            if tr_deps < 0.0 {
                l * density_1 * (1.5 * l * tr_deps.powi(2) + c * 0.06 * tr_deps.abs())
            } else {
                0.0
            }
        };
        output.set_scalar(Q::BulkViscosity, ip, q_1);
        // /***********************************************************************
        //  * Combine deviatoric and volumetric stresses
        //  **********************************************************************/
        let s_1 = s_tr * alpha;
        let sigma_1 = s_1 - MANDEL_IDENTITY * (p_1 + q_1);
        output.set_vector(Q::MandelStress, ip, sigma_1);

        // ***********************************************************************
        // Update optional output variables if needed
        // **********************************************************************

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
            output.set_scalar(Q::Pressure, ip, p_1);
        }

        let elastic_rate =
            -(1. - alpha) / (2. * self.parameters.shear_modulus * del_t) * s_0 + alpha * d_eps_dev;

        let density_mid = 0.5 * (density_0 + density_1);
        if output.is_some(Q::InternalPlasticEnergy) && input.is_some(Q::InternalPlasticEnergy) {
            let s_mid = 0.5 * (s_0 + s_1);
            let deviatoric_rate = d_eps_dev - elastic_rate;
            let e_0 = input.get_scalar(Q::InternalPlasticEnergy, ip);
            let e_1 = e_0 + del_t / density_mid * (s_mid.dot(&deviatoric_rate));
            output.set_scalar(Q::InternalPlasticEnergy, ip, e_1);
        }
        if output.is_some(Q::InternalElasticEnergy) && input.is_some(Q::InternalElasticEnergy) {
            let s_mid = 0.5 * (s_0 + s_1);
            let p_mid = -0.5 * (p_0 + p_1);
            let deviatoric_rate = elastic_rate;
            let e_0 = input.get_scalar(Q::InternalElasticEnergy, ip);
            let e_1 =
                e_0 + del_t / density_mid * (s_mid.dot(&deviatoric_rate) + 3. * d_eps_vol * p_mid);
            output.set_scalar(Q::InternalElasticEnergy, ip, e_1);
        }
        if output.is_some(Q::InternalEnergy) && input.is_some(Q::InternalEnergy) {
            let e_0 = input.get_scalar(Q::InternalEnergy, ip);
            let sigma_mid = 0.5 * (sigma_0 + sigma_1);
            let e_1 = e_0 + del_t / density_mid * sigma_mid.dot(&d_eps);
            output.set_scalar(Q::InternalEnergy, ip, e_1);
        }
        if output.is_some(Q::InternalHeatingEnergy) && input.is_some(Q::InternalHeatingEnergy) {
            let e_0 = input.get_scalar(Q::InternalHeatingEnergy, ip);
            let q_mid = -0.5 * (q_1 + input.get_scalar(Q::BulkViscosity, ip));
            let e_1 = e_0 + del_t / density_mid * 3. * q_mid * d_eps_vol;
            output.set_scalar(Q::InternalHeatingEnergy, ip, e_1);
        }
        if output.is_some(Q::EqPlasticStrain) && input.is_some(Q::EqPlasticStrain) {
            output.set_scalar(
                Q::EqPlasticStrain,
                ip,
                input.get_scalar(Q::EqPlasticStrain, ip) + del_lambda,
            );
        }
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::VelocityGradient, QDim::SquareTensor(3)),
            (Q::CellDiameter, QDim::Scalar),
        ])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions. These Variables are
    /// stored both in in the input and the output.
    fn define_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
            (Q::Damage, QDim::Scalar),
            (Q::BulkingPressure, QDim::Scalar),
            (Q::Density, QDim::Scalar),
            (Q::BulkViscosity, QDim::Scalar),
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
        HashMap::from([
            (Q::EqStrainRate, QDim::Scalar),
            (Q::MandelStrainRate, QDim::Vector(6)),
            (Q::MisesStress, QDim::Scalar),
            (Q::Pressure, QDim::Scalar),
        ])
    }
    fn define_optional_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::InternalPlasticEnergy, QDim::Scalar),
            (Q::InternalElasticEnergy, QDim::Scalar),
            (Q::InternalEnergy, QDim::Scalar),
            (Q::InternalHeatingEnergy, QDim::Scalar),
        ])
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("density".to_string(), self.parameters.density),
            ("shear_modulus".to_string(), self.parameters.shear_modulus),
            ("bulk_modulus".to_string(), self.parameters.bulk_modulus),
            ("a_y".to_string(), self.parameters.a_y),
            ("b_y".to_string(), self.parameters.b_y),
            ("d_y".to_string(), self.parameters.d_y),
            ("a_r".to_string(), self.parameters.a_r),
            ("b_r".to_string(), self.parameters.b_r),
            ("e_f".to_string(), self.parameters.e_f),
        ])
    }
}
