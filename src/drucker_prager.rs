use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    mandel_decomposition, mandel_rate_from_velocity_gradient, I_6, MANDEL_IDENTITY,
    PROJECTION_DEV_6,
};
use nalgebra::{RowSVector, SMatrix, SMatrixView, SVector};
use strum_macros::Display;

use core::{f64, panic};
use std::collections::HashMap;
use std::convert::identity;
use std::error::Error;
use std::fmt::Debug;

pub trait IsotropicHardeningPlasticity3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self>
    where
        Self: Sized;
    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, 6>,
        sigma_1: &SVector<f64, 6>,
        del_eps: &SVector<f64, 6>,
        kappa: f64,
        nonlocal_strain: f64,
        damage_0: f64,
    );
    fn damage(&self) -> f64;
    fn f(&self) -> f64;
    fn df_dsigma(&self) -> SVector<f64, 6>;
    fn df_dkappa(&self) -> f64;
    //fn g(&self) -> f64;
    fn m(&self) -> SVector<f64, 6>;
    fn dm_dkappa(&self) -> SVector<f64, 6>;
    fn dm_dsigma(&self) -> SMatrix<f64, 6, 6>;
    fn k(&self) -> f64;
    fn dk_dsigma(&self) -> SVector<f64, 6>;
    fn dk_dkappa(&self) -> f64;
    fn elastic_tangent(&self) -> SMatrixView<f64, 6, 6>;
    fn elastic_tangent_inv(&self) -> SMatrixView<f64, 6, 6>;
    fn del_plastic_strain(&self) -> SVector<f64, 6>;
}

#[derive(Debug, Default)]
struct DruckerPragerState {
    i_1: f64,
    j_2: f64,
    stress_dev: SVector<f64, 6>,
    del_plastic_strain: SVector<f64, 6>,
    history: f64,
    nonlocal_strain: f64,
    damage: f64,
    f: f64,
    df_dsigma: SVector<f64, 6>,
    df_dkappa: f64,
    m: SVector<f64, 6>,
    dm_dsigma: SMatrix<f64, 6, 6>,
    dm_dkappa: SVector<f64, 6>,
    k: f64,
    dk_dsigma: SVector<f64, 6>,
    dk_dkappa: f64,
}
#[derive(Debug, Default)]
pub struct DruckerPrager3D {
    pub a_y: f64,
    pub b_y: f64,
    pub d_y: f64,
    pub e_f: f64,
    pub h: f64,
    pub alpha_0: f64,
    pub radial_factor: f64, //for one: pure radial return
    D: SMatrix<f64, 6, 6>,
    D_inv: SMatrix<f64, 6, 6>,
    state: DruckerPragerState,
}
impl IsotropicHardeningPlasticity3D for DruckerPrager3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self>
    where
        Self: Sized,
    {
        let shear_modulus = *parameters.get("shear_modulus")?;
        let bulk_modulus = *parameters.get("bulk_modulus")?;
        let lambda = bulk_modulus - 2.0 * shear_modulus / 3.0;
        let D = PROJECTION_DEV_6 * lambda + SMatrix::<f64, 6, 6>::identity() * 2. * lambda;
        let D_inv = D.try_inverse()?;
        Some(Self {
            a_y: *parameters.get("a_y")?,
            b_y: *parameters.get("b_y")?,
            d_y: *parameters.get("d_y")?,
            e_f: *parameters.get("e_f")?,
            h: *parameters.get("h")?,
            alpha_0: *parameters.get("alpha_0")?,
            radial_factor: *parameters.get("radial_factor")?,
            D: D,
            D_inv: D_inv,
            state: DruckerPragerState::default(),
        })
    }
    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, 6>,
        sigma_1: &SVector<f64, 6>,
        del_eps: &SVector<f64, 6>,
        kappa: f64,
        nonlocal_strain: f64,
        damage_0: f64,
    ) {
        let (p, s) = mandel_decomposition(sigma_1);
        let i_1 = -3.0 * p;
        let j_2 = 0.5 * s.norm_squared();
        self.state.i_1 = i_1;
        self.state.j_2 = j_2;
        self.state.stress_dev = s;
        self.state.history = kappa;
        self.state.nonlocal_strain = nonlocal_strain;
        self.state.del_plastic_strain = del_eps - self.D_inv * (sigma_1 - sigma_0);
        //println!("del_pl: {}", self.state.del_plastic_strain);
        self.state.damage = {
            let damage_1 = 1. - f64::exp((self.alpha_0 - nonlocal_strain) / self.e_f);
            if damage_1 > damage_0 {
                damage_1
            } else {
                damage_0
            }
        };
        self.state.f = self.state.j_2
            - (1. - self.state.damage)
                * (1. + self.h * self.state.history)
                * (self.b_y / self.a_y).powi(2)
                * ((self.state.i_1 - self.d_y).powi(2) - self.a_y.powi(2));
        assert!(!self.state.f.is_nan(), "f is NaN");
        assert!(!self.state.f.is_infinite(), "f is infinite");
        //println!("f: {}", self.state.f);
        //println!("i_1: {}", self.state.i_1);
        //println!("j_2: {}", self.state.j_2);
        //println!("kappa: {}", kappa);
        let df_di_1 = -2.0
            * (1. - self.state.damage)
            * (1. + self.h * self.state.history)
            * (self.b_y / self.a_y).powi(2)
            * (self.state.i_1 - self.d_y);

        let df_dj_2 = 1.0;
        self.state.df_dkappa = -(1. - self.state.damage)
            * self.h
            * (self.b_y / self.a_y).powi(2)
            * ((self.state.i_1 - self.d_y).powi(2) - self.a_y.powi(2));

        self.state.df_dsigma = df_di_1 * MANDEL_IDENTITY + s;
        self.state.m = (1.0 - self.radial_factor) * df_di_1 * MANDEL_IDENTITY + s;

        let dm_di_1 = -2.0
            * (1. - self.radial_factor)
            * (1. - self.state.damage)
            * (1. + self.h * self.state.history)
            * (self.b_y / self.a_y).powi(2);
        self.state.dm_dsigma =
            MANDEL_IDENTITY * dm_di_1 * MANDEL_IDENTITY.transpose() + PROJECTION_DEV_6;
        self.state.dm_dkappa = -2.0
            * (1. - self.radial_factor)
            * (1. - self.state.damage)
            * self.h
            * (self.b_y / self.a_y).powi(2)
            * (self.state.i_1 - self.d_y)
            * MANDEL_IDENTITY;
        let pl_norm = self.state.del_plastic_strain.norm();
        self.state.k = f64::sqrt(2. / 3.) * pl_norm;
        self.state.dk_dsigma = {
            if pl_norm == 0.0 {
                SVector::<f64, 6>::zeros()
            } else {
            - f64::sqrt(2. / 3.) * self.D_inv * self.state.del_plastic_strain / pl_norm
            }
        };
        self.state.dk_dkappa = 0.0;
    }

    fn damage(&self) -> f64 {
        self.state.damage
    }
    fn f(&self) -> f64 {
        self.state.f
    }
    fn df_dsigma(&self) -> SVector<f64, 6> {
        self.state.df_dsigma
    }
    fn df_dkappa(&self) -> f64 {
        self.state.df_dkappa
    }
    fn m(&self) -> SVector<f64, 6> {
        self.state.m
    }
    fn dm_dsigma(&self) -> SMatrix<f64, 6, 6> {
        self.state.dm_dsigma
    }
    fn dm_dkappa(&self) -> SVector<f64, 6> {
        self.state.dm_dkappa
    }
    fn k(&self) -> f64 {
        self.state.k
    }
    fn dk_dsigma(&self) -> SVector<f64, 6> {
        self.state.dk_dsigma
    }
    fn dk_dkappa(&self) -> f64 {
        self.state.dk_dkappa
    }
    fn elastic_tangent(&self) -> SMatrixView<f64, 6, 6> {
        SMatrixView::from(&self.D)
    }
    fn elastic_tangent_inv(&self) -> SMatrixView<f64, 6, 6> {
        SMatrixView::from(&self.D_inv)
    }
    fn del_plastic_strain(&self) -> SVector<f64, 6> {
        self.state.del_plastic_strain
    }
}
#[derive(Debug, Default)]
pub struct DruckerPrager23D {
    pub a_y: f64,
    pub b_y: f64,
    pub d_y: f64,
    pub e_f: f64,
    pub h: f64,
    pub alpha_0: f64,
    pub radial_factor: f64, //for one: pure radial return
    D: SMatrix<f64, 6, 6>,
    D_inv: SMatrix<f64, 6, 6>,
    state: DruckerPragerState,
}
impl IsotropicHardeningPlasticity3D for DruckerPrager23D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self>
    where
        Self: Sized,
    {
        let shear_modulus = *parameters.get("shear_modulus")?;
        let bulk_modulus = *parameters.get("bulk_modulus")?;
        let lambda = bulk_modulus - 2.0 * shear_modulus / 3.0;
        let D = PROJECTION_DEV_6 * lambda + SMatrix::<f64, 6, 6>::identity() * 2. * lambda;
        let D_inv = D.try_inverse()?;
        Some(Self {
            a_y: *parameters.get("a_y")?,
            b_y: *parameters.get("b_y")?,
            d_y: *parameters.get("d_y")?,
            e_f: *parameters.get("e_f")?,
            h: *parameters.get("h")?,
            alpha_0: *parameters.get("alpha_0")?,
            radial_factor: *parameters.get("radial_factor")?,
            D: D,
            D_inv: D_inv,
            state: DruckerPragerState::default(),
        })
    }
    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, 6>,
        sigma_1: &SVector<f64, 6>,
        del_eps: &SVector<f64, 6>,
        kappa: f64,
        nonlocal_strain: f64,
        damage_0: f64,
    ) {
        let (p, s) = mandel_decomposition(sigma_1);
        let i_1 = -3.0 * p;
        let j_2 = 0.5 * s.norm_squared();
        self.state.i_1 = i_1;
        self.state.j_2 = j_2;
        self.state.stress_dev = s;
        self.state.history = kappa;
        self.state.nonlocal_strain = nonlocal_strain;
        self.state.del_plastic_strain = del_eps - self.D_inv * (sigma_1 - sigma_0);
        //println!("del_pl: {}", self.state.del_plastic_strain);
        self.state.damage = {
            let damage_1 = 1. - f64::exp((self.alpha_0 - nonlocal_strain) / self.e_f);
            if damage_1 > damage_0 {
                damage_1
            } else {
                damage_0
            }
        };
        let b = (1.0 - self.state.damage) * (1. + self.h * self.state.history) * self.b_y;
        let db_dkappa = (1. - self.state.damage) * self.h * self.b_y;

        self.state.f =
            self.state.i_1 + self.a_y * (self.state.j_2 + b.powi(2)).sqrt() / b - self.d_y;
        assert!(!self.state.f.is_nan(), "f is NaN");
        assert!(!self.state.f.is_infinite(), "f is infinite");
        //println!("f: {}", self.state.f);
        //println!("i_1: {}", self.state.i_1);
        //println!("j_2: {}", self.state.j_2);
        //println!("kappa: {}", kappa);
        let df_di_1 = 1.0;

        let df_dj_2 = (1_f64 / 2.0) * self.a_y / ((self.state.j_2 + b.powi(2)).sqrt() * b);

        self.state.df_dkappa = -self.a_y * (self.state.j_2 + b.powi(2)).sqrt() * db_dkappa
            / b.powi(2)
            + self.a_y * db_dkappa / (self.state.j_2 + b.powi(2)).sqrt();

        self.state.df_dsigma =  MANDEL_IDENTITY + df_dj_2 * s;
        self.state.m = (1.0 - self.radial_factor) * MANDEL_IDENTITY + df_dj_2 * s;

        let df_di_1i_1 = 0.0;
        let df_dj_2j_2 = -1_f64/4.0*self.a_y/((self.state.j_2 + b.powi(2)).powf(3_f64/2.0)*b);
        self.state.dm_dsigma =
            s * df_dj_2j_2 * s.transpose() + df_dj_2 * PROJECTION_DEV_6;
        let df_dj_2kappa = -1_f64/2.0*self.a_y*db_dkappa/((self.state.j_2 + b.powi(2)).sqrt()*b.powi(2)) - 1_f64/2.0*self.a_y*db_dkappa/(self.state.j_2 + b.powi(2)).powf(3_f64/2.0);
        self.state.dm_dkappa = df_dj_2kappa * s;
        let pl_norm = self.state.del_plastic_strain.norm();
        self.state.k = f64::sqrt(2. / 3.) * pl_norm;
        self.state.dk_dsigma = {
            if pl_norm == 0.0 {
                SVector::<f64, 6>::zeros()
            } else {
                - f64::sqrt(2. / 3.) * self.D_inv * self.state.del_plastic_strain / pl_norm
            }
        };
        self.state.dk_dkappa = 0.0;
    }

    fn damage(&self) -> f64 {
        self.state.damage
    }
    fn f(&self) -> f64 {
        self.state.f
    }
    fn df_dsigma(&self) -> SVector<f64, 6> {
        self.state.df_dsigma
    }
    fn df_dkappa(&self) -> f64 {
        self.state.df_dkappa
    }
    fn m(&self) -> SVector<f64, 6> {
        self.state.m
    }
    fn dm_dsigma(&self) -> SMatrix<f64, 6, 6> {
        self.state.dm_dsigma
    }
    fn dm_dkappa(&self) -> SVector<f64, 6> {
        self.state.dm_dkappa
    }
    fn k(&self) -> f64 {
        self.state.k
    }
    fn dk_dsigma(&self) -> SVector<f64, 6> {
        self.state.dk_dsigma
    }
    fn dk_dkappa(&self) -> f64 {
        self.state.dk_dkappa
    }
    fn elastic_tangent(&self) -> SMatrixView<f64, 6, 6> {
        SMatrixView::from(&self.D)
    }
    fn elastic_tangent_inv(&self) -> SMatrixView<f64, 6, 6> {
        SMatrixView::from(&self.D_inv)
    }
    fn del_plastic_strain(&self) -> SVector<f64, 6> {
        self.state.del_plastic_strain
    }
}
#[derive(Debug, Default)]
pub struct DruckerPragerClassic3D {
    pub a_y: f64,
    pub b_y: f64,
    pub d_y: f64,
    pub e_f: f64,
    pub h: f64,
    pub alpha_0: f64,
    pub radial_factor: f64, //for one: pure radial return
    D: SMatrix<f64, 6, 6>,
    D_inv: SMatrix<f64, 6, 6>,
    state: DruckerPragerState,
}
impl IsotropicHardeningPlasticity3D for DruckerPragerClassic3D {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self>
    where
        Self: Sized,
    {
        let shear_modulus = *parameters.get("shear_modulus")?;
        let bulk_modulus = *parameters.get("bulk_modulus")?;
        let lambda = bulk_modulus - 2.0 * shear_modulus / 3.0;
        let D = PROJECTION_DEV_6 * lambda + SMatrix::<f64, 6, 6>::identity() * 2. * lambda;
        let D_inv = D.try_inverse()?;
        Some(Self {
            a_y: *parameters.get("a_y")?,
            b_y: *parameters.get("b_y")?,
            d_y: *parameters.get("d_y")?,
            e_f: *parameters.get("e_f")?,
            h: *parameters.get("h")?,
            alpha_0: *parameters.get("alpha_0")?,
            radial_factor: *parameters.get("radial_factor")?,
            D: D,
            D_inv: D_inv,
            state: DruckerPragerState::default(),
        })
    }
    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, 6>,
        sigma_1: &SVector<f64, 6>,
        del_eps: &SVector<f64, 6>,
        kappa: f64,
        nonlocal_strain: f64,
        damage_0: f64,
    ) {
        let (p, s) = mandel_decomposition(sigma_1);
        let i_1 = -3.0 * p;
        let j_2 = 0.5 * s.norm_squared();
        self.state.i_1 = i_1;
        self.state.j_2 = j_2;
        self.state.stress_dev = s;
        self.state.history = kappa;
        self.state.nonlocal_strain = nonlocal_strain;
        self.state.del_plastic_strain = del_eps - self.D_inv * (sigma_1 - sigma_0);
        //println!("del_pl: {}", self.state.del_plastic_strain);
        self.state.damage = {
            let damage_1 = 1. - f64::exp((self.alpha_0 - nonlocal_strain) / self.e_f);
            if damage_1 > damage_0 {
                damage_1
            } else {
                damage_0
            }
        };
        let b = (1.0 - self.state.damage) * (1. + self.h * self.state.history) * self.b_y;
        let db_dkappa = (1. - self.state.damage) * self.h * self.b_y;

        self.state.f = self.state.j_2.sqrt()+ (b/self.a_y)*self.state.i_1-b/self.a_y*self.d_y;
        assert!(!self.state.f.is_nan(), "f is NaN");
        assert!(!self.state.f.is_infinite(), "f is infinite");
        //println!("f: {}", self.state.f);
        //println!("i_1: {}", self.state.i_1);
        //println!("j_2: {}", self.state.j_2);
        //println!("kappa: {}", kappa);
        let df_di_1 = b/self.a_y;

        let df_dj_2 = 0.5 *(1.0/ self.state.j_2.sqrt());

        self.state.df_dkappa = (db_dkappa/self.a_y)*self.state.i_1-db_dkappa/self.a_y*self.d_y;

        self.state.df_dsigma = df_di_1 * MANDEL_IDENTITY + df_dj_2 * s;
        self.state.m = (1.0 - self.radial_factor) * df_di_1 * MANDEL_IDENTITY + df_dj_2 * s;

        let df_di_1i_1 = 0.0;
        let df_dj_2j_2 = -0.25*(1.0/(self.state.j_2.sqrt().powi(3)));
        self.state.dm_dsigma =
            s * df_dj_2j_2 * s.transpose() + df_dj_2 * PROJECTION_DEV_6;
        let df_dj_2kappa = 0.0;
        let df_di_1kappa = db_dkappa/self.a_y;
        self.state.dm_dkappa =(1.0 - self.radial_factor) * df_di_1kappa * MANDEL_IDENTITY;
        let pl_norm = self.state.del_plastic_strain.norm();
        self.state.k = f64::sqrt(2. / 3.) * pl_norm;
        self.state.dk_dsigma = {
            if pl_norm == 0.0 {
                SVector::<f64, 6>::zeros()
            } else {
                -f64::sqrt(2. / 3.) * self.D_inv * self.state.del_plastic_strain / pl_norm
            }
        };
        self.state.dk_dkappa = 0.0;
    }

    fn damage(&self) -> f64 {
        self.state.damage
    }
    fn f(&self) -> f64 {
        self.state.f
    }
    fn df_dsigma(&self) -> SVector<f64, 6> {
        self.state.df_dsigma
    }
    fn df_dkappa(&self) -> f64 {
        self.state.df_dkappa
    }
    fn m(&self) -> SVector<f64, 6> {
        self.state.m
    }
    fn dm_dsigma(&self) -> SMatrix<f64, 6, 6> {
        self.state.dm_dsigma
    }
    fn dm_dkappa(&self) -> SVector<f64, 6> {
        self.state.dm_dkappa
    }
    fn k(&self) -> f64 {
        self.state.k
    }
    fn dk_dsigma(&self) -> SVector<f64, 6> {
        self.state.dk_dsigma
    }
    fn dk_dkappa(&self) -> f64 {
        self.state.dk_dkappa
    }
    fn elastic_tangent(&self) -> SMatrixView<f64, 6, 6> {
        SMatrixView::from(&self.D)
    }
    fn elastic_tangent_inv(&self) -> SMatrixView<f64, 6, 6> {
        SMatrixView::from(&self.D_inv)
    }
    fn del_plastic_strain(&self) -> SVector<f64, 6> {
        self.state.del_plastic_strain
    }
}
#[derive(Debug)]
pub struct Plasticity3D<MODEL: IsotropicHardeningPlasticity3D + Debug> {
    density: f64,
    shear_modulus: f64,
    bulk_modulus: f64,
    model: MODEL,
}

impl<MODEL: IsotropicHardeningPlasticity3D + Debug> ConstitutiveModel for Plasticity3D<MODEL> {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            density: *parameters.get("density")?,
            shear_modulus: *parameters.get("shear_modulus")?,
            bulk_modulus: *parameters.get("bulk_modulus")?,
            model: MODEL::new(parameters)?,
        })
    }
    fn evaluate_ip(
        &mut self,
        ip: usize,
        del_t: f64,
        input: &QValueInput,
        output: &mut QValueOutput,
    ) {
        let velocity_gradient = input
            .get_tensor::<{ Q::VelocityGradient.dim() }, { Q::VelocityGradient.size() }>(
                Q::VelocityGradient,
                ip,
            );
        let d_eps = mandel_rate_from_velocity_gradient(&velocity_gradient);
        let (mut d_eps_vol, d_eps_dev) = mandel_decomposition(&d_eps);
        d_eps_vol *= -1.0;

        let sigma_0 = input.get_vector::<{ Q::MandelStress.size() }>(Q::MandelStress, ip);
        //let mut sigma_1 = SVector::<f64, 6>::zeros();
        let alpha_0 = input.get_scalar(Q::EqPlasticStrain, ip);
        let mut alpha_1 = alpha_0;
        let nonlocal_strain = input.get_scalar(Q::EqNonlocalPlasticStrain, ip);
        //let mut history = alpha_0;
        let damage_0 = input.get_scalar(Q::Damage, ip);
        let mut damage_1: f64;
        let (p_0, s_0) = mandel_decomposition(&sigma_0);
        let p_0 = - (p_0 - input.get_scalar(Q::BulkViscosity, ip));

        let sigma_tr = s_0
            + 2. * self.shear_modulus * d_eps_dev * del_t
            + MANDEL_IDENTITY * (p_0 + self.bulk_modulus * 3.0 * d_eps_vol * del_t);
        let mut sigma_1 = sigma_tr;
        self.model.set_model_state(
            &sigma_0,
            &sigma_1,
            &(d_eps * del_t),
            alpha_0,
            nonlocal_strain,
            damage_0,
        );

        if self.model.f() < 0.0 {
            // No plasticity
            //sigma_1 = sigma_tr;
            alpha_1 = alpha_0;
            damage_1 = self.model.damage();
        } else {
            let mut del_lambda = 0.0;
            let mut sol_0 = SVector::<f64, 8>::from_element(0.0); //make first test fail
            let mut sol_1 = SVector::<f64, 8>::from([
                sigma_tr[0],
                sigma_tr[1],
                sigma_tr[2],
                sigma_tr[3],
                sigma_tr[4],
                sigma_tr[5],
                alpha_0,
                0.0,
            ]);
            let mut res_sigma =
                sigma_tr - sigma_0 - del_lambda * self.model.elastic_tangent() * self.model.m();
            let mut res_kappa = alpha_1 - alpha_0 - self.model.k();
            let mut res_f = self.model.f();

            let mut dres = SMatrix::<f64, 8, 8>::zeros();

            let mut res = SVector::<f64, 8>::from([
                res_sigma[0],
                res_sigma[1],
                res_sigma[2],
                res_sigma[3],
                res_sigma[4],
                res_sigma[5],
                res_kappa,
                res_f,
            ]);
            let mut i = 0;
            let maxit = 100;
            //println!("residual: {}", res);
            let res_sigma_0_norm = res_sigma.norm();
            let res_kappa_0_norm = res_kappa.abs();
            let res_f_0_norm = res_f.abs();
            let mut sigma_prev = sigma_0;
            let mut alpha_prev = sol_0[6];
            let mut del_lambda_prev = sol_0[7];
            while (res_sigma.norm() > 1e-6 * res_sigma_0_norm
                || res_kappa.abs() > 1e-6 * res_kappa_0_norm
                || res_f.abs() > 1e-6 * res_f_0_norm)
                || ((sigma_1 - sigma_prev).norm() > 1e-6 * sigma_1.norm()
                    || (alpha_1 - alpha_prev).abs() > 1e-6 * alpha_1.abs()
                    || (del_lambda - del_lambda_prev).abs() > 1e-6 * del_lambda.abs())
            {
                sol_0 = sol_1;

                // fill dres_sigma_dsigma
                dres.fixed_view_mut::<6, 6>(0, 0).copy_from(
                    &(-I_6 - self.model.elastic_tangent() * del_lambda * self.model.dm_dsigma()),
                );
                //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
                dres.fixed_view_mut::<6, 1>(0, 6).copy_from(
                    &(-self.model.elastic_tangent() * del_lambda * self.model.dm_dkappa()),
                );
                //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
                dres.fixed_view_mut::<6, 1>(0, 7)
                    .copy_from(&(-self.model.elastic_tangent() * self.model.m()));

                //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
                dres.fixed_view_mut::<1, 6>(6, 0)
                    .copy_from_slice((-self.model.dk_dsigma()).as_slice());
                //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
                dres.fixed_view_mut::<1, 1>(6, 6)
                    .copy_from_slice(&[1.0 - self.model.dk_dkappa()]);
                //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
                dres.fixed_view_mut::<1, 1>(6, 7).copy_from_slice(&[0.0]);

                //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
                dres.fixed_view_mut::<1, 6>(7, 0)
                    .copy_from(&self.model.df_dsigma().transpose());
                //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
                dres.fixed_view_mut::<1, 1>(7, 6)
                    .copy_from_slice(&[self.model.df_dkappa()]);
                //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
                dres.fixed_view_mut::<1, 1>(7, 7).copy_from_slice(&[0.0]);

                let lu = dres.lu();
                let result = lu.solve(&res);
                match result {
                    Some(sol) => sol_1 = sol_0 - sol,
                    None => {
                        println!("model {:#?}", self.model);
                        panic!(
                            "Plasticity3D: Newton-Raphson failed to solve system {}",
                            dres
                        );
                    }
                }

                // extract solution and calcualte new residual
                sigma_1 = sol_1.fixed_view::<6, 1>(0, 0).into();
                alpha_1 = sol_1[6];
                del_lambda = sol_1[7];

                sigma_prev = sol_0.fixed_view::<6, 1>(0, 0).into();
                alpha_prev = sol_0[6];
                del_lambda_prev = sol_0[7];

                self.model.set_model_state(
                    &sigma_0,
                    &sigma_1,
                    &(d_eps * del_t),
                    alpha_1,
                    nonlocal_strain,
                    damage_0,
                );

                res_sigma =
                    sigma_tr - sigma_1 - del_lambda * self.model.elastic_tangent() * self.model.m();
                res_kappa = alpha_1 - alpha_0 - self.model.k();
                res_f = self.model.f();
                res = SVector::<f64, 8>::from([
                    res_sigma[0],
                    res_sigma[1],
                    res_sigma[2],
                    res_sigma[3],
                    res_sigma[4],
                    res_sigma[5],
                    res_kappa,
                    res_f,
                ]);
                if i > maxit {
                    panic!("Plasticity3D: Newton-Raphson did not converge. residual: {}, solution change: {}", res.norm(), (sol_1 - sol_0).norm() / sol_1.norm());
                }
                i += 1;
            }
            damage_1 = self.model.damage();
            println!("del_alpha = {}", alpha_1-alpha_0);
            //println!("final f: {}", self.model.f());
        }
        output.set_scalar(Q::EqPlasticStrain, ip, alpha_1);
        output.set_scalar(Q::Damage, ip, damage_1);
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

        let mu = density_1 / self.density - 1.;

        // Calculate bulk viscosity
        let l = input.get_scalar(Q::CellDiameter, ip);
        let tr_deps = d_eps_vol * 3.;
        let c = (self.bulk_modulus / self.density).sqrt();
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
        output.set_vector(Q::MandelStress, ip, sigma_1 - q_1 * MANDEL_IDENTITY);

        // ***********************************************************************
        // Update optional output variables if needed
        // **********************************************************************

        if output.is_some(Q::MandelStrainRate) {
            output.set_vector(Q::MandelStrainRate, ip, d_eps);
        }

        let density_mid = 0.5 * (density_0 + density_1);
        if output.is_some(Q::InternalPlasticEnergy) && input.is_some(Q::InternalPlasticEnergy) {
            let sigma_mid = 0.5 * (sigma_0 + sigma_1);
            let e_0 = input.get_scalar(Q::InternalPlasticEnergy, ip);
            let e_1 = e_0 + 1. / density_mid * (sigma_mid.dot(&self.model.del_plastic_strain()));
            output.set_scalar(Q::InternalPlasticEnergy, ip, e_1);
        }
        if output.is_some(Q::InternalElasticEnergy) && input.is_some(Q::InternalElasticEnergy) {
            let sigma_mid = 0.5 * (sigma_0 + sigma_1);
            let e_0 = input.get_scalar(Q::InternalPlasticEnergy, ip);
            let del_elastic_strain = d_eps * del_t - self.model.del_plastic_strain();
            let e_1 = e_0 + 1. / density_mid * (sigma_mid.dot(&del_elastic_strain));
            output.set_scalar(Q::InternalPlasticEnergy, ip, e_1);
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
    }

    /// Returns the physical quantities that are required as input for the
    /// constitutive model together with their dimensions.
    fn define_input(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::VelocityGradient, QDim::SquareTensor(3)),
            (Q::CellDiameter, QDim::Scalar),
            (Q::EqNonlocalPlasticStrain, QDim::Scalar),
            (Q::MandelStress, QDim::Vector(6)),
        ])
    }

    /// Returns the physical quantities that are needed as internal variables
    /// for the constitutive model together with their dimensions. These Variables are
    /// stored both in in the input and the output.
    fn define_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::EqPlasticStrain, QDim::Scalar),
            (Q::Damage, QDim::Scalar),
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
        HashMap::from([(Q::MandelStrainRate, QDim::Vector(6))])
    }
    fn define_optional_history(&self) -> HashMap<Q, QDim> {
        HashMap::from([
            (Q::InternalPlasticEnergy, QDim::Scalar),
            (Q::InternalElasticEnergy, QDim::Scalar),
            (Q::InternalEnergy, QDim::Scalar),
            (Q::InternalHeatingEnergy, QDim::Scalar),
        ])
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("density".to_string(), self.density),
            ("shear_modulus".to_string(), self.shear_modulus),
            ("bulk_modulus".to_string(), self.bulk_modulus),
        ])
    }
}
