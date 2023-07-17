use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{tensor_to_mandel, mandel_to_tensor, T_dev, T_vol, T_id};

use nalgebra::{SMatrix, default_allocator};
use std::collections::HashMap;
use std::cmp;

struct JH2Parameters {
    RHO:f64,
    SHEAR_MODULUS:f64,
    A:f64,
    B:f64,
    C:f64,
    M:f64,
    N:f64,
    EPS0:f64,
    T:f64,
    SIGMAHEL:f64,
    PHEL:f64,
    D1:f64,
    D2:f64,
    K1:f64,
    K2:f64,
    K3:f64,
    BETA:f64,
    EFMIN:f64, 
}
pub struct JH23D {
    parameters: JH2Parameters,
}

impl JH23D {
    pub fn new(parameters: &HashMap<String, f64>) -> Self {
        let RHO = parameters.get("RHO").unwrap();
        let SHEAR_MODULUS = parameters.get("SHEAR_MODULUS").unwrap();
        let A = parameters.get("A").unwrap();
        let B = parameters.get("B").unwrap();
        let C = parameters.get("C").unwrap();
        let M = parameters.get("M").unwrap();
        let N = parameters.get("N").unwrap();
        let EPS0 = parameters.get("EPS0").unwrap();
        let T = parameters.get("T").unwrap();
        let SIGMAHEL = parameters.get("SIGMAHEL").unwrap();
        let PHEL = parameters.get("PHEL").unwrap();
        let D1 = parameters.get("D1").unwrap();
        let D2 = parameters.get("D2").unwrap();
        let K1 = parameters.get("K1").unwrap();
        let K2 = parameters.get("K2").unwrap();
        let K3 = parameters.get("K3").unwrap();
        let BETA = parameters.get("BETA").unwrap();
        let EFMIN = parameters.get("EFMIN").unwrap();
        Self { parameters: JH2Parameters {
            RHO: *RHO,
            SHEAR_MODULUS: *SHEAR_MODULUS,
            A: *A,
            B: *B,
            C: *C,
            M: *M,
            N: *N,
            EPS0: *EPS0,
            T: *T,
            SIGMAHEL: *SIGMAHEL,
            PHEL: *PHEL,
            D1: *D1,
            D2: *D2,
            K1: *K1,
            K2: *K2,
            K3: *K3,
            BETA: *BETA,
            EFMIN: *EFMIN,
        } }
    }
}

impl ConstitutiveModel for JH23D {
    fn evaluate_ip(&self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let L = SMatrix::<f64, 3,3>::from_row_slice(&input.get_slice::<9>(Q::VelocityGradient, ip));
        let D = 0.5 * (L + L.transpose());
        
        let d_eps = tensor_to_mandel(D);
        // //const auto d_eps_vol = T_vol.dot(d_eps);
        
        let sigma_0 = input.get_vector::<6>(Q::MandelStress, ip); 

        
        let lambda_0 = input.get_scalar(Q::Lambda, ip);
        // //const double e_n = _internal_vars[E](i);
        let damage_0 = input.get_scalar(Q::Damage, ip);
        let mut damage_1 = damage_0;

        
        //let sigma_1 = mandel_to_tensor(sigma_0);
        // //stress += h * (stress * W_.transpose() + W_ * stress);

        // /***********************************************************************
        //  * START CONSTITUTIVE MODEL HERE
        //  * 1) Calculate failure surface Y_failure
        //  * 2) Calculate Yield surface Y_yield = f(Y_failure)
        //  **********************************************************************/
        let p_0 = - T_vol.dot(&sigma_0);
        let s_0 = T_dev * sigma_0;
        let s_tr = s_0 + 2. * self.parameters.SHEAR_MODULUS * (T_dev * d_eps) * del_t;
        let s_tr_eq = (1.5 * s_tr.norm_squared()).sqrt();
        let d_eps_eq = ((2./3.) * d_eps.norm_squared()).sqrt();
        let mut alpha = 0.0;

        let p_s = p_0 / self.parameters.PHEL;
        let t_s = self.parameters.T / self.parameters.PHEL;
        //let del_lambda = 0.0;
        //let fracture_surface = 0.0;
        let residual_surface = 0.0;
        let mut rate_factor = 1.;


        let fracture_surface = (self.parameters.A * (p_s + t_s).powf(self.parameters.N) * self.parameters.SIGMAHEL).max(0.0);
        let residual_surface = (self.parameters.B * (p_s).powf(self.parameters.M) * self.parameters.SIGMAHEL).max(0.0);
        if d_eps_eq >= self.parameters.EPS0{
            rate_factor += self.parameters.C * (d_eps_eq/self.parameters.EPS0).ln();
        } 
        let yield_surface = {
            if damage_0 == 0.0{
                fracture_surface*rate_factor
            } else {
                (fracture_surface*(1.-damage_0) + damage_0 * fracture_surface)*rate_factor
            }
        };
        if s_tr_eq > yield_surface {
            let e_p_f = (self.parameters.D1 * (p_s + t_s).powf(self.parameters.D2)).max(self.parameters.EFMIN);
            
            let del_lambda =(s_tr_eq-yield_surface) / (3.*self.parameters.SHEAR_MODULUS);
            alpha = yield_surface/s_tr_eq;

            output.set_scalar(Q::Lambda, ip, lambda_0 + del_lambda);
            // Update damage variable or set to 1.
            damage_1 = (damage_0+del_lambda/e_p_f).min(1.0);
            output.set_scalar(Q::Damage, ip, damage_1);
        
        } else {
            //elastic
            alpha = 1.0;
        }

        // /***********************************************************************
        //  * END CONSTITUTIVE MODEL HERE
        //  **********************************************************************/

        // /***********************************************************************
        //  * UPDATE DENSITY
        //  * The density is updated using the explicit midpoint rule for the
        //  * deformation gradient.
        //  **********************************************************************/
        
        let f1 = del_t/2. * L.trace();
        let density_1 = input.get_scalar(Q::Density, ip) * (1.+f1)/(1.-f1);
        output.set_scalar(Q::Density, ip, density_1);
        // /***********************************************************************
        //  * UPDATE ENERGY AND EOS
        //  **********************************************************************/
        
        let mu = density_1/self.parameters.RHO -1.;
        
        let mut p: f64;// = (mu > 0) ? _param->K1 * mu + _param->K2 * mu * mu + _param->K3 * mu * mu * mu + _internal_vars[Q::PRESSURE](i): _param->K1 * mu;
        if mu > 0.0{
            p = self.parameters.K1 * mu + self.parameters.K2 * mu.powi(2) + self.parameters.K3 * mu.powi(3) + input.get_scalar(Q::Pressure, ip);
        } else {
            p = (self.parameters.K1 * mu).max(-self.parameters.T * (1.-damage_1));
        }
        if damage_1 > damage_0 && self.parameters.BETA > 0.0{
            let y_old = (damage_0 * residual_surface + (1.-damage_0) * fracture_surface);
            let y_new = (damage_1 * residual_surface + (1.-damage_1) * fracture_surface);
            let u_old = (y_old * y_old) / (6. * self.parameters.SHEAR_MODULUS);
            let u_new = (y_new * y_new) / (6. * self.parameters.SHEAR_MODULUS);

            let del_u = u_old - u_new;
            
            let del_p_n = input.get_scalar(Q::Pressure, ip) - p;
            let del_p = -self.parameters.K1 * mu + ((self.parameters.K1 * mu + del_p_n).powi(2)+2.*self.parameters.BETA * self.parameters.K1 * del_u).sqrt();
            output.set_scalar(Q::Pressure, ip, del_p);
        }

        // /***********************************************************************
        //  * Combine deviatoric and volumetric stresses and apply stress rate
        //  **********************************************************************/

        output.set_vector(Q::MandelStress, ip, alpha * s_tr - T_id * p); 

    }
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::VelocityGradient, QDim::Tensor(3,3)),
            (Q::MandelStress, QDim::Vector(6)),
            (Q::Damage, QDim::Scalar),
            (Q::Lambda, QDim::Scalar),
            (Q::Pressure, QDim::Scalar),
            (Q::Density, QDim::Scalar),
            ])
    }
    fn define_output(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::MandelStress, QDim::Vector(6)),
            (Q::Damage, QDim::Scalar),
            (Q::Lambda, QDim::Scalar),
            (Q::Pressure, QDim::Scalar),
            (Q::Density, QDim::Scalar),
        ])
    }
}
