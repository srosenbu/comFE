use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    mandel_decomposition, mandel_rate_from_velocity_gradient, MANDEL_IDENTITY,
};

use std::collections::HashMap;
#[derive(Debug)]
pub struct MisesPlasticity3D {
    mu: f64,
    kappa: f64,
    sigma_y: f64,
    h: f64,
}

impl ConstitutiveModel for MisesPlasticity3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            mu: *parameters.get("mu").unwrap(),
            kappa: *parameters.get("kappa").unwrap(),
            sigma_y: *parameters.get("sigma_y").unwrap(),
            h: *parameters.get("h").unwrap(),
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

        let (p_0, s_0) = mandel_decomposition(&sigma_0);
        let p_1 = p_0 - self.kappa * 3.0 * d_eps_vol * del_t;

        let s_tr = s_0 + 2. * self.mu * d_eps_dev * del_t;
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();

        let sigma_y = self.sigma_y + self.h * lambda_0;

        //the .max(0.0) contains the check if the stress is already above the yield surface
        let del_lambda = ((s_tr_eq - sigma_y) / (3. * self.mu + self.h)).max(0.0);
        let alpha = 1. - (3. * self.mu * del_lambda) / s_tr_eq;
        let s_1 = alpha * s_tr;
        let sigma_1 = s_1 - MANDEL_IDENTITY * p_1;
        output.set_vector(Q::MandelStress, ip, sigma_1);
        output.set_scalar(Q::EqPlasticStrain, ip, lambda_0 + del_lambda);

        let elastic_rate = -(1. - alpha) / (2. * self.mu * del_t) * s_0 + alpha * d_eps_dev;

        let f1 = del_t / 2. * 3. * d_eps_vol;
        let density_0 = input.get_scalar(Q::Density, ip);
        let density_1 = density_0 * (1. - f1) / (1. + f1);
        output.set_scalar(Q::Density, ip, density_1);
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
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([(Q::VelocityGradient, QDim::SquareTensor(3))])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions. These Variables are
    /// stored both in in the input and the output.
    fn define_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::Density, QDim::Scalar),
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
        HashMap::from([
            (Q::InternalPlasticEnergy, QDim::Scalar),
            (Q::InternalElasticEnergy, QDim::Scalar),
            (Q::InternalEnergy, QDim::Scalar),
        ])
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("mu".to_string(), self.mu),
            ("kappa".to_string(), self.kappa),
            ("sigma_y".to_string(), self.sigma_y),
            ("h".to_string(), self.h),
        ])
    }
}

#[derive(Debug)]
pub struct MisesPlasticityExponentialSoftening3D {
    mu: f64,
    kappa: f64,
    sigma_y: f64,
    e_f: f64,
    density: f64,
    //alpha_0: f64,
}

impl ConstitutiveModel for MisesPlasticityExponentialSoftening3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            mu: *parameters.get("mu").unwrap(),
            kappa: *parameters.get("kappa").unwrap(),
            sigma_y: *parameters.get("sigma_y").unwrap(),
            e_f: *parameters.get("e_f").unwrap(),
            density: *parameters.get("density").unwrap(),
            //alpha_0: *parameters.get("alpha_0").unwrap(),
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

        let (p_0, s_0) = mandel_decomposition(&sigma_0);
        let p_1 = p_0 - self.kappa *3.0* d_eps_vol * del_t;

        let s_tr = s_0 + 2. * self.mu * d_eps_dev * del_t;
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();

        let mut sigma_y = self.sigma_y * (f64::exp(-lambda_0 / self.e_f));
        //the .max(0.0) contains the check if the stress is already above the yield surface
        let mut del_lambda = 0.0;
        let mut lambda_1 = lambda_0;
        let alpha;
        if s_tr_eq > sigma_y {
            let mut iter = 0;
            let mut dsigma_y = 0.0;
            loop{
                sigma_y = self.sigma_y * (f64::exp(-lambda_1 / self.e_f));
                dsigma_y = self.sigma_y * (-1. / self.e_f) * f64::exp(-lambda_1 / self.e_f);
                lambda_1 = lambda_1 - (sigma_y - s_tr_eq + 3.*self.mu*del_lambda) / (3. * self.mu + dsigma_y);
                del_lambda = lambda_1 - lambda_0;
                iter += 1;
                if del_lambda.abs()/lambda_1.abs() < 1e-6 || iter > 100 {
                    break;
                }
            }
            alpha = 1. - (3. * self.mu * del_lambda) / s_tr_eq;
        } else {
            alpha = 1.0;
        }

        let s_1 = alpha * s_tr;
        let sigma_1 = s_1 - MANDEL_IDENTITY * p_1;
        output.set_vector(Q::MandelStress, ip, sigma_1);
        output.set_scalar(Q::EqPlasticStrain, ip, lambda_1);
        output.set_scalar(Q::Damage, ip, 1. - f64::exp(-lambda_1 / self.e_f));

        let elastic_rate = -(1. - alpha) / (2. * self.mu * del_t) * s_0 + alpha * d_eps_dev;

        let f1 = del_t / 2. * 3. * d_eps_vol;
        let density_0 = input.get_scalar(Q::Density, ip);
        let density_1 = density_0 * (1. - f1) / (1. + f1);
        output.set_scalar(Q::Density, ip, density_1);
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
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([(Q::VelocityGradient, QDim::SquareTensor(3))])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions. These Variables are
    /// stored both in in the input and the output.
    fn define_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::Density, QDim::Scalar),
            (Q::Damage, QDim::Scalar),
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
        HashMap::from([
            (Q::InternalPlasticEnergy, QDim::Scalar),
            (Q::InternalElasticEnergy, QDim::Scalar),
            (Q::InternalEnergy, QDim::Scalar),
        ])
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("mu".to_string(), self.mu),
            ("kappa".to_string(), self.kappa),
            ("sigma_y".to_string(), self.sigma_y),
            ("e_f".to_string(), self.e_f),
            ("density".to_string(), self.density),
        ])
    }
}
#[derive(Debug)]
pub struct UniaxialStressMisesPlasticityExponentialSoftening3D {
    mu: f64,
    kappa: f64,
    sigma_y: f64,
    e_f: f64,
    density: f64,
    //alpha_0: f64,
}

impl ConstitutiveModel for UniaxialStressMisesPlasticityExponentialSoftening3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            mu: *parameters.get("mu").unwrap(),
            kappa: *parameters.get("kappa").unwrap(),
            sigma_y: *parameters.get("sigma_y").unwrap(),
            e_f: *parameters.get("e_f").unwrap(),
            density: *parameters.get("density").unwrap(),
            //alpha_0: *parameters.get("alpha_0").unwrap(),
        })
    }
    fn evaluate_ip(&mut self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let velocity_gradient = input
            .get_tensor::<{ Q::VelocityGradient.dim() }, { Q::VelocityGradient.size() }>(
                Q::VelocityGradient,
                ip,
            );
        let mut d_eps = mandel_rate_from_velocity_gradient(&velocity_gradient);
        let youngs_modulus = 9. * self.mu * self.kappa / (self.mu + 3.*self.kappa);
        //let poisson_ratio = youngs_modulus / (2. * self.mu) - 1.0;
        //d_eps.y = - poisson_ratio * d_eps.x;
        //d_eps.z = - poisson_ratio * d_eps.x;

        //let (mut d_eps_vol, d_eps_dev) = mandel_decomposition(&d_eps);
        //d_eps_vol *= -1.0;
        
        let sigma_0 = input.get_vector::<{ Q::MandelStress.size() }>(Q::MandelStress, ip);
        let alpha_0 = input.get_scalar(Q::EqPlasticStrain, ip);

        //let (p_0, s_0) = mandel_decomposition(&sigma_0);
        //let p_1 = p_0 - self.kappa * d_eps_vol * del_t;
        let sigma_tr = youngs_modulus * &d_eps * del_t + &sigma_0;
        let (p_tr, s_tr) = mandel_decomposition(&sigma_tr);
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();

        let mut sigma_y = self.sigma_y * (f64::exp(-alpha_0 / self.e_f));
        //the .max(0.0) contains the check if the stress is already above the yield surface
        let mut del_alpha = 0.0;
        let mut alpha_1 = alpha_0;
        let mut alpha_1_old:f64;
        let mut f:f64;
        let mut df:f64;
        let theta;
        if s_tr_eq > sigma_y {
            let mut iter = 0;
            let mut dsigma_y:f64;
            loop{
                alpha_1_old = alpha_1;
                sigma_y = self.sigma_y * (f64::exp(-alpha_1 / self.e_f));
                dsigma_y = self.sigma_y * (-1. / self.e_f) * f64::exp(-alpha_1 / self.e_f);
                f = sigma_y - s_tr_eq + 3.*self.mu*del_alpha;
                df = dsigma_y + 3.*self.mu;
                alpha_1 = alpha_1 - f/df;
                del_alpha = alpha_1 - alpha_0;
                
                iter += 1;
                //println!("iter: {}, lambda_1: {}, e/l1: {}, f: {}", iter, lambda_1, (lambda_1-lambda_1_old).abs()/lambda_1.abs(), f);
                if (alpha_1-alpha_1_old).abs()/alpha_1.abs() < 1e-5 || f.abs() < 1e-8 {
                    break;
                } else if iter > 100 {
                    println!("sigma_y_0: {}, sigma_y: {}, s_tr_eq: {}, lambda_0: {}, lambda_1: {}", self.sigma_y*(1.-f64::exp(-alpha_0/self.e_f)) ,sigma_y, s_tr_eq, alpha_0, alpha_1);
                    panic!("Plasticity iteration did not converge");
                }
            }
            assert!(alpha_1 > 0.0, "lambda_1: {} not greater than zero", alpha_1);
            theta = 1. - (3. * self.mu * del_alpha) / s_tr_eq;
        } else {
            theta = 1.0;
        }
        let n = s_tr/s_tr_eq;
        let del_gamma = f64::sqrt(3. / 2.) * del_alpha;
        let del_eps_pl = del_gamma * &n;

        let s_1 = theta * s_tr;
        let sigma_11 = 1.5 * s_1;
        let p_1 = 0.5*(s_1.y+s_1.z); //s_1.y and s_1.z are equal
        let mut sigma_1 = s_1;
        output.set_vector(Q::MandelStress, ip, sigma_1);
        output.set_scalar(Q::EqPlasticStrain, ip, alpha_1);
        output.set_scalar(Q::Damage, ip, 1. - f64::exp(-alpha_1 / self.e_f));

        let elastic_rate = -(1. - theta) / (2. * self.mu * del_t) * s_0 + theta * d_eps_dev;

        let f1 = del_t / 2. * 3. * d_eps_vol;
        let density_0 = input.get_scalar(Q::Density, ip);
        let density_1 = density_0 * (1. - f1) / (1. + f1);
        output.set_scalar(Q::Density, ip, density_1);
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
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([(Q::VelocityGradient, QDim::SquareTensor(3))])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions. These Variables are
    /// stored both in in the input and the output.
    fn define_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::Density, QDim::Scalar),
            (Q::Damage, QDim::Scalar),
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
        HashMap::from([
            (Q::InternalPlasticEnergy, QDim::Scalar),
            (Q::InternalElasticEnergy, QDim::Scalar),
            (Q::InternalEnergy, QDim::Scalar),
        ])
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("mu".to_string(), self.mu),
            ("kappa".to_string(), self.kappa),
            ("sigma_y".to_string(), self.sigma_y),
            ("e_f".to_string(), self.e_f),
            ("density".to_string(), self.density),
        ])
    }
}
