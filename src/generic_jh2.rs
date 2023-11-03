use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    deviatoric, mandel_decomposition, mandel_rate_from_velocity_gradient, MANDEL_IDENTITY,
};

use nalgebra::{SMatrix, DVectorView};
use std::cmp;
use std::collections::HashMap;
pub enum Parameter<'a> {
    Constant(f64),
    Function(DVectorView::<'a, f64>),
}
impl Parameter<'_> {
    fn value(&self, i: usize) -> f64 {
        match self {
            Parameter::Constant(value) => *value,
            Parameter::Function(function) => function[i],
        }
    }
}
pub trait JH2Parameters {
    fn from_hash_map(parameters: &HashMap<String, Parameter>) -> Option<Self> where Self: Sized;
    fn rho(&self, i: usize) -> f64;
    fn shear_modulus(&self, i: usize) -> f64;
    fn a(&self, i: usize) -> f64;
    fn b(&self, i: usize) -> f64;
    fn c(&self, i: usize) -> f64;
    fn m(&self, i: usize) -> f64;
    fn n(&self, i: usize) -> f64;
    fn eps0(&self, i: usize) -> f64;
    fn t(&self, i: usize) -> f64;
    fn sigmahel(&self, i: usize) -> f64;
    fn phel(&self, i: usize) -> f64;
    fn d1(&self, i: usize) -> f64;
    fn d2(&self, i: usize) -> f64;
    fn k1(&self, i: usize) -> f64;
    fn k2(&self, i: usize) -> f64;
    fn k3(&self, i: usize) -> f64;
    fn beta(&self, i: usize) -> f64;
    fn efmin(&self, i: usize) -> f64;
    fn dmax(&self, i: usize) -> f64;
}

pub struct GenericJH23D<P: JH2Parameters> {
    parameters: P,
}

impl<P: JH2Parameters> GenericJH23D::<P> {
    fn try_from_map(parameters: &HashMap<String, Parameter>) -> Option<Self> {
        let new = Self {
            parameters: P::from_hash_map(parameters)?,
        };
        Some(new)
        
    }
    fn evaluate_ip(&self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let velocity_gradient = input
            .get_tensor::<{ Q::VelocityGradient.dim() }, { Q::VelocityGradient.size() }>(
                Q::VelocityGradient,
                ip,
            );
        let d_eps = mandel_rate_from_velocity_gradient(&velocity_gradient);
        let (mut d_eps_vol, d_eps_dev) = mandel_decomposition(&d_eps);
        d_eps_vol *= -1.0;

        let sigma_0 = input.get_vector::<{ Q::MandelStress.size() }>(Q::MandelStress, ip);

        let mut del_lambda = 0.0;

        let damage_0 = input.get_scalar(Q::Damage, ip);
        let mut damage_1 = damage_0;

        let (p_0, s_0) = mandel_decomposition(&sigma_0);
        let s_tr = s_0 + 2. * self.parameters.shear_modulus(ip) * d_eps_dev * del_t;
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();
        let d_eps_eq = ((2. / 3.) * d_eps.norm_squared()).sqrt();
        let mut alpha;

        let p_s = p_0 / self.parameters.phel(ip);
        let t_s = self.parameters.t(ip) / self.parameters.phel(ip);
        let mut rate_factor = 1.;

        let fracture_surface =
            (self.parameters.a(ip) * (p_s + t_s).powf(self.parameters.n(ip)) * self.parameters.sigmahel(ip))
                .max(0.0);
        let residual_surface =
            (self.parameters.b(ip) * (p_s).powf(self.parameters.m(ip)) * self.parameters.sigmahel(ip)).max(0.0);
        if d_eps_eq >= self.parameters.eps0(ip) {
            rate_factor += self.parameters.c(ip) * (d_eps_eq / self.parameters.eps0(ip)).ln();
        }
        let yield_surface = rate_factor * {
            if damage_0 == 0.0 {
                fracture_surface
            } else {
                fracture_surface * (1. - damage_0) + damage_0 * residual_surface
            }
        };
        if s_tr_eq > yield_surface {
            let e_p_f = (self.parameters.d1(ip) * (p_s + t_s).powf(self.parameters.d2(ip)))
                .max(self.parameters.efmin(ip));

            del_lambda = (s_tr_eq - yield_surface) / (3. * self.parameters.shear_modulus(ip));
            alpha = yield_surface / s_tr_eq;

            damage_1 = (damage_0 + del_lambda / e_p_f).min(self.parameters.dmax(ip));
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
        assert!(density_1 > 0.0, "Negative density encountered in JH2 model: {}", density_1);
        output.set_scalar(Q::Density, ip, density_1);

        let mu = density_1 / self.parameters.rho(ip) - 1.;
        
        //let mut del_p_0 = 0.0;
        let mut d_eps_vol_pl = 0.0;
        let p_1 = {
            if mu > 0.0 {
                self.parameters.k1(ip) * mu
                    + self.parameters.k2(ip) * mu.powi(2)
                    + self.parameters.k3(ip) * mu.powi(3)
                    + input.get_scalar(Q::BulkingPressure, ip)
            } else {
                let p_trial = self.parameters.k1(ip)*mu;
                let p_damaged = -self.parameters.t(ip) * (1. - damage_1);
                if p_trial > p_damaged {
                    p_trial
                } else {
                    //let denominator = (density_1-density_0)/self.parameters.RHO;
                    let pl = p_damaged - p_0;
                    let el = p_trial - p_0;
                    d_eps_vol_pl = {
                        if el != 0.0 {
                            (1. - pl / el) * d_eps_vol
                        } else {
                            d_eps_vol
                        }
                    };
                    p_damaged
                }
            }
        };
        if damage_1 > damage_0 {
            let y_old = damage_0 * residual_surface + (1. - damage_0) * fracture_surface;
            let y_new = damage_1 * residual_surface + (1. - damage_1) * fracture_surface;
            let u_old = y_old.powi(2) / (6. * self.parameters.shear_modulus(ip));
            let u_new = y_new.powi(2) / (6. * self.parameters.shear_modulus(ip));

            let del_u = u_old - u_new;

            let del_p_0 = input.get_scalar(Q::BulkingPressure, ip);
            let del_p = -self.parameters.k1(ip) * mu
                + ((self.parameters.k1(ip) * mu + del_p_0).powi(2)
                    + 2. * self.parameters.beta(ip) * self.parameters.k1(ip) * del_u)
                    .sqrt();
            output.set_scalar(Q::BulkingPressure, ip, del_p);
        }

        // /***********************************************************************
        //  * Combine deviatoric and volumetric stresses
        //  **********************************************************************/
        let s_1 = s_tr * alpha;
        let sigma_1 = s_1 - MANDEL_IDENTITY * p_1;
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
        //if output.is_some(Q::InternalEnergyRate) {
        //    output.set_scalar(Q::InternalEnergyRate, ip, sigma_1.dot(&d_eps));
        //}
        
        
        // Update optional internal variables if needed

        if output.is_some(Q::InternalPlasticEnergy) && input.is_some(Q::InternalPlasticEnergy) {
            let s_mid = 0.5 * (s_0 + s_1);
            let p_mid = - 0.5 * (p_0 + p_1); 
            let density_mid = 0.5 * (density_0 + density_1);
            let deviatoric_rate = d_eps_dev * (1.-alpha);
            let e_0 = input.get_scalar(Q::InternalPlasticEnergy, ip);
            let e_1 = e_0 + del_t/density_mid * (s_mid.dot(&deviatoric_rate) + 3. * d_eps_vol_pl * p_mid);
            output.set_scalar(Q::InternalPlasticEnergy, ip, e_1);
        }
        if output.is_some(Q::InternalElasticEnergy) && input.is_some(Q::InternalElasticEnergy) {
            let s_mid = 0.5 * (s_0 + s_1);
            let p_mid = - 0.5 * (p_0 + p_1); 
            let density_mid = 0.5 * (density_0 + density_1);
            let deviatoric_rate = d_eps_dev * alpha;
            let e_0 = input.get_scalar(Q::InternalElasticEnergy, ip);
            let e_1 = e_0 + del_t/density_mid * (s_mid.dot(&deviatoric_rate) + 3. * (d_eps_vol - d_eps_vol_pl) * p_mid);
            output.set_scalar(Q::InternalElasticEnergy, ip, e_1);
        }
        if output.is_some(Q::InternalEnergy) && input.is_some(Q::InternalEnergy) {
            let e_0 = input.get_scalar(Q::InternalEnergy, ip);
            let sigma_mid = 0.5 * (sigma_0 + sigma_1);
            let density_mid = 0.5 * (density_0 + density_1);
            let e_1 = e_0 + del_t/density_mid * sigma_mid.dot(&d_eps);
            output.set_scalar(Q::InternalEnergy, ip, e_1);
        }
        if output.is_some(Q::EqPlasticStrain) && input.is_some(Q::EqPlasticStrain) {
            output.set_scalar(Q::EqPlasticStrain, ip, input.get_scalar(Q::EqPlasticStrain, ip) + del_lambda);
        }
            
        
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
            (Q::EqStrainRate, QDim::Scalar),
            (Q::MandelStrainRate, QDim::Vector(6)),
            (Q::MisesStress, QDim::Scalar),
            (Q::Pressure, QDim::Scalar),
            //(Q::InternalEnergyRate, QDim::Scalar),
            //(Q::InternalElasticEnergyRate, QDim::Scalar),
            //(Q::InternalPlasticEnergyRate, QDim::Scalar),
        ])
    }
    fn define_optional_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::InternalPlasticEnergy, QDim::Scalar),
            (Q::InternalElasticEnergy, QDim::Scalar),
            (Q::InternalEnergy, QDim::Scalar),
        ])
    }

}
