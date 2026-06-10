use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::stress_strain::{
    mandel_decomposition, mandel_rate_from_velocity_gradient, I_6, MANDEL_IDENTITY,
    PROJECTION_DEV_6,
};
use nalgebra::{ComplexField, RowSVector, SMatrix, SMatrixView, SVector};
use strum_macros::Display;

use core::{f64, panic};
use std::collections::HashMap;
use std::convert::identity;
use std::error::Error;
use std::fmt::Debug;

fn projection_matrix(n: SVector<f64, 3>) -> SMatrix<f64, 6, 3> {
    let (n1, n2, n3) = (n[0], n[1], n[2]);
    let factor = 1.0 / std::f64::consts::SQRT_2;
    SMatrix::<f64, 6, 3>::from_row_slice(&[
        n1,
        0.0,
        0.0,
        0.0,
        n2,
        0.0,
        0.0,
        0.0,
        n3,
        0.0,
        n3 * factor,
        n2 * factor,
        n3 * factor,
        0.0,
        n1 * factor,
        n2 * factor,
        n1 * factor,
        0.0,
    ])
}

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
    fn update_newton_matrix<const N: usize>(&self, dres: &mut SMatrix<f64, N, N>, del_lambda: f64) {
        const STRESS_STRAIN: usize = 6;
        const KAPPA: usize = 1;
        assert!(N == STRESS_STRAIN + KAPPA + 1);
        // fill dres_sigma_dsigma
        dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
            .copy_from(
                &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                    + self.elastic_tangent() * del_lambda * self.dm_dsigma()),
            );
        //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
        dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
            .copy_from(&(self.elastic_tangent() * self.m()));
        //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
        dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
            .copy_from(&(self.elastic_tangent() * del_lambda * self.dm_dkappa()));

        //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
        dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
            .copy_from(&self.df_dsigma().transpose());
        //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
        dres[(STRESS_STRAIN, STRESS_STRAIN)] = 0.0;
        //.fixed_view_mut::<1, 1>(STRESS_STRAIN, STRESS_STRAIN)
        //    .copy_from_slice(&[0.0]);
        //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
        dres[(STRESS_STRAIN, STRESS_STRAIN + 1)] = self.df_dkappa();
        //dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN+1)
        //    .copy_from(&SVector::<f64,1>::from_element(self.df_dkappa()));

        //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
        dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
            .copy_from(&((-del_lambda) * self.dk_dsigma().transpose()));
        //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
        dres[(STRESS_STRAIN + 1, STRESS_STRAIN)] = -self.k();
        //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
        dres[(STRESS_STRAIN + 1, STRESS_STRAIN + 1)] = 1.0 - del_lambda * self.dk_dkappa();
    }
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
    pub a_r: f64,
    pub b_r: f64,
    pub d_r: f64,
    pub e_f: f64,
    pub h: f64,
    pub h_a: f64,
    pub h_d: f64,
    pub alpha_0: f64,
    pub radial_factor: f64,  //for one: pure radial return
    pub omega_max: f64,
    //pub tensile_cutoff: f64, //for one: tensile cutoff
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
        let D = SMatrix::<f64, 6, 6>::new(
            2. * shear_modulus + lambda,
            lambda,
            lambda,
            0.,
            0.,
            0.,
            lambda,
            2. * shear_modulus + lambda,
            lambda,
            0.,
            0.,
            0.,
            lambda,
            lambda,
            2. * shear_modulus + lambda,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            2. * shear_modulus,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            2. * shear_modulus,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            2. * shear_modulus,
        );
        let D_inv = D.try_inverse()?;
        Some(Self {
            a_y: *parameters.get("a_y")?,
            b_y: *parameters.get("b_y")?,
            d_y: *parameters.get("d_y")?,
            a_r: *parameters.get("a_r")?,
            b_r: *parameters.get("b_r")?,
            d_r: *parameters.get("d_r")?,
            e_f: *parameters.get("e_f")?,
            h: *parameters.get("h")?,
            h_a: *parameters.get("h_a").unwrap_or(parameters.get("h")?),
            h_d: *parameters.get("h_d").unwrap_or(parameters.get("h")?),
            alpha_0: *parameters.get("alpha_0")?,
            radial_factor: *parameters.get("radial_factor")?,
            omega_max: *parameters.get("omega_max").unwrap_or(&1.0),
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
        self.state.damage = {
            let damage_1 = self.omega_max*(1. - f64::exp((self.alpha_0 - nonlocal_strain) / self.e_f));
            if damage_1 > damage_0 {
                damage_1
            } else {
                damage_0
            }
        };
        let (p, s) = mandel_decomposition(sigma_1);
        let i_1 = -3.0 * p;
        let j_2 = 0.5 * s.norm_squared();
        self.state.i_1 = i_1;
        self.state.j_2 = j_2;
        self.state.stress_dev = s;
        self.state.history = kappa;
        self.state.nonlocal_strain = nonlocal_strain;
        self.state.del_plastic_strain = del_eps - self.D_inv * (sigma_1 - sigma_0);

        let b = (1.0 + self.h * self.state.history) * (1.0 - self.state.damage) * self.b_y
            + self.state.damage * self.b_r;
        let a = (1.0 + self.h_a * self.state.history) * (1.0 - self.state.damage) * self.a_y
            + self.state.damage * self.a_r;
        let d = (1.0 + self.h_d * self.state.history) * (1.0 - self.state.damage) * self.d_y
            + self.state.damage * self.d_r;
        let db_dkappa = self.h * (1.0 - self.state.damage) * self.b_y;
        let da_dkappa = self.h_a * (1.0 - self.state.damage) * self.a_y;
        let dd_dkappa = self.h_d * (1.0 - self.state.damage) * self.d_y;

        self.state.f = self.state.i_1 + a * (self.state.j_2 + b.powi(2)).sqrt() / b - d;
        assert!(!self.state.f.is_nan(), "f is NaN");
        assert!(!self.state.f.is_infinite(), "f is infinite");
        //println!("f: {}", self.state.f);
        //println!("i_1: {}", self.state.i_1);
        //println!("j_2: {}", self.state.j_2);
        //println!("kappa: {}", kappa);
        let df_di_1 = 1.0;

        let df_dj_2 = (1.0 / 2.0) * a / ((self.state.j_2 + b.powi(2)).sqrt() * b);

        self.state.df_dkappa = -(self.state.j_2 + b.powi(2)).sqrt() * a * db_dkappa / b.powi(2)
            + (self.state.j_2 + b.powi(2)).sqrt() * da_dkappa / b
            - dd_dkappa
            + a * db_dkappa / (self.state.j_2 + b.powi(2)).sqrt();

        self.state.df_dsigma = &MANDEL_IDENTITY + df_dj_2 * &s;
        self.state.m = (1.0 - self.radial_factor) * &MANDEL_IDENTITY + df_dj_2 * &s;
        let df_di_1i_1 = 0.0;
        let df_dj_2j_2 =
            -1_f64 / 4.0 * a / ((self.state.j_2 + b.powi(2)).powf(3_f64 / 2.0) * b);
        self.state.dm_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV_6;
        let df_dj_2kappa = -1.0 / 2.0 * a * db_dkappa
            / ((self.state.j_2 + b.powi(2)).sqrt() * b.powi(2))
            + (1.0 / 2.0) * da_dkappa / ((self.state.j_2 + b.powi(2)).sqrt() * b)
            - 1.0 / 2.0 * a * db_dkappa / (self.state.j_2 + b.powi(2)).powf(3.0 / 2.0);
        self.state.dm_dkappa = df_dj_2kappa * s;

        let pl_norm = self.state.del_plastic_strain.norm();

        let m_norm = self.state.m.norm();
        self.state.k = (2_f64 / 3_f64).sqrt() * m_norm;
        self.state.dk_dsigma =
            (((2_f64 / 3_f64).sqrt() / m_norm) * self.state.m.transpose() * &self.state.dm_dsigma)
                .transpose();
        self.state.dk_dkappa = (((2_f64 / 3_f64).sqrt() / m_norm)
            * self.state.m.transpose()
            * &self.state.dm_dkappa)[(0, 0)];
        //self.state.k = f64::sqrt(2. / 3.) * pl_norm;
        // self.state.dk_dsigma = {
        //     if pl_norm == 0.0 {
        //         SVector::<f64, 6>::zeros()
        //     } else {
        //         -f64::sqrt(2. / 3.) * self.D_inv * self.state.del_plastic_strain / pl_norm
        //     }
        // };
        // self.state.dk_dkappa = 0.0;
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
        let D = SMatrix::<f64, 6, 6>::new(
            2. * shear_modulus + lambda,
            lambda,
            lambda,
            0.,
            0.,
            0.,
            lambda,
            2. * shear_modulus + lambda,
            lambda,
            0.,
            0.,
            0.,
            lambda,
            lambda,
            2. * shear_modulus + lambda,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            2. * shear_modulus,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            2. * shear_modulus,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            2. * shear_modulus,
        );
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
            self.state.j_2.sqrt() + (b / self.a_y) * self.state.i_1 - b / self.a_y * self.d_y;
        assert!(!self.state.f.is_nan(), "f is NaN");
        assert!(!self.state.f.is_infinite(), "f is infinite");

        let df_di_1 = b / self.a_y;

        let df_dj_2 = 0.5 * (1.0 / self.state.j_2.sqrt());

        self.state.df_dkappa =
            (db_dkappa / self.a_y) * self.state.i_1 - db_dkappa / self.a_y * self.d_y;

        self.state.df_dsigma = df_di_1 * MANDEL_IDENTITY + df_dj_2 * s;
        self.state.m = (1.0 - self.radial_factor) * df_di_1 * MANDEL_IDENTITY + df_dj_2 * s;

        let df_di_1i_1 = 0.0;
        let df_dj_2j_2 = -0.25 * (1.0 / (self.state.j_2.sqrt().powi(3)));
        self.state.dm_dsigma = s * df_dj_2j_2 * s.transpose() + df_dj_2 * PROJECTION_DEV_6;
        let df_dj_2kappa = 0.0;
        let df_di_1kappa = db_dkappa / self.a_y;
        self.state.dm_dkappa = (1.0 - self.radial_factor) * df_di_1kappa * MANDEL_IDENTITY;
        let pl_norm = self.state.del_plastic_strain.norm();
        //FIXME: The hardening evolution is wrong here
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
    n1: f64,
    n2: f64,
    n3: f64,
    model: MODEL,
}

impl<MODEL: IsotropicHardeningPlasticity3D + Debug> ConstitutiveModel for Plasticity3D<MODEL> {
    fn new(parameters: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            density: *parameters.get("density")?,
            shear_modulus: *parameters.get("shear_modulus")?,
            bulk_modulus: *parameters.get("bulk_modulus")?,
            n1: *parameters.get("n1").unwrap_or(&1.0),
            n2: *parameters.get("n2").unwrap_or(&1.0),
            n3: *parameters.get("n3").unwrap_or(&1.0),
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
        let p_0 = -(p_0 - input.get_scalar(Q::BulkViscosity, ip));

        let sigma_tr = self.model.elastic_tangent() * (del_t * d_eps) + sigma_0;
        let mut sigma_1 = sigma_tr;
        self.model.set_model_state(
            &sigma_0,
            &sigma_1,
            &(d_eps * del_t),
            alpha_0,
            nonlocal_strain,
            damage_0,
        );
        let (n1, n2, n3) = (self.n1, self.n2, self.n3);
        if self.model.f() < 0.0 {
            // No plasticity
            //sigma_1 = sigma_tr;
            //println!(self.model.del_plastic_strain().norm())
            alpha_1 = alpha_0;
            damage_1 = self.model.damage();
            if output.is_some(Q::StabilityEigenvalue) {
                let tangent = self.model.elastic_tangent().symmetric_part();
                let P = projection_matrix(SVector::<f64, 3>::new(n1, n2, n3));
                let eig = (P.transpose() * &tangent * &P).symmetric_eigenvalues();
                let out = eig.min();
                output.set_scalar(Q::StabilityEigenvalue, ip, out);
            }
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
                0.0,
                alpha_0,
            ]);

            let mut res_sigma = SVector::<f64, 6>::zeros();
            let mut res_kappa = 0.0;
            let mut res_f = self.model.f();

            let mut dres = SMatrix::<f64, 8, 8>::zeros();

            let mut res = SVector::<f64, 8>::from([
                res_sigma[0],
                res_sigma[1],
                res_sigma[2],
                res_sigma[3],
                res_sigma[4],
                res_sigma[5],
                res_f,
                res_kappa,
            ]);
            let mut i = 0;
            let maxit = 25;
            let atol = 1e-8;
            let rtol = 1e-8;
            let mut sigma_prev: SVector<f64, 6>;
            let mut alpha_prev: f64;
            let mut del_lambda_prev: f64;

            // start the newton-raphson iteration. The condition is checked at the end
            // of the loop to ensure at least one iteration.
            loop {
                sol_0 = sol_1;

                // fill dres_sigma_dsigma
                self.model.update_newton_matrix(&mut dres, del_lambda);

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
                alpha_1 = sol_1[7];
                del_lambda = sol_1[6];

                sigma_prev = sol_0.fixed_view::<6, 1>(0, 0).into();
                alpha_prev = sol_0[7];
                del_lambda_prev = sol_0[6];

                self.model.set_model_state(
                    &sigma_0,
                    &sigma_1,
                    &(d_eps * del_t),
                    alpha_1,
                    nonlocal_strain,
                    damage_0,
                );

                res_sigma =
                    sigma_1 - sigma_tr + del_lambda * self.model.elastic_tangent() * self.model.m();
                res_kappa = alpha_1 - alpha_0 - del_lambda * self.model.k();
                res_f = self.model.f();
                res = SVector::<f64, 8>::from([
                    res_sigma[0],
                    res_sigma[1],
                    res_sigma[2],
                    res_sigma[3],
                    res_sigma[4],
                    res_sigma[5],
                    res_f,
                    res_kappa,
                ]);
                if res_sigma.norm() < atol && res_kappa.abs() < atol && res_f.abs() < atol {
                    break;
                }
                if (sigma_1 - sigma_prev).norm() < atol + rtol * sigma_1.norm()
                    && (alpha_1 - alpha_prev).abs() < atol + rtol * alpha_1.abs()
                    && (del_lambda - del_lambda_prev).abs() < atol + rtol * del_lambda.abs()
                {
                    break;
                }
                if i > maxit {
                    panic!("Plasticity3D: Newton-Raphson did not converge. residual: {}, solution change: {}", res.norm(), (sol_1 - sol_0).norm() / sol_1.norm());
                }
                i += 1;
            }
            assert!(
                del_lambda > 0.0,
                "non-positive plastic multiplier{}",
                del_lambda
            );
            damage_1 = self.model.damage();

            if output.is_some(Q::StabilityEigenvalue) {
                self.model.update_newton_matrix(&mut dres, del_lambda);

                let inv = dres.try_inverse().expect("Tangent evaluation failed");
                let tangent =
                    (&inv.fixed_view::<6, 6>(0, 0) * self.model.elastic_tangent()).symmetric_part();
                if n2 == 0.0 && n3 == 0.0 {
                    let P = projection_matrix(SVector::<f64, 3>::new(n1, n2, n3));
                    let eig = (P.transpose() * &tangent * &P).symmetric_eigenvalues();
                    let out = eig.min();
                    output.set_scalar(Q::StabilityEigenvalue, ip, out);
                } else if n2.abs()>0.0 && n3==0.0 {
                    // parametrice half unit circle with the angle theta
                    let theta:SVector<f64, 10> = SVector::<f64, 10>::from_iterator((0..10).map(|i| (i as f64) * std::f64::consts::FRAC_PI_2 / 9.0)); 
                    // calculate the eigenvalue for each direction and take the minimum
                    let mut min_eig = f64::INFINITY;
                    for i in 0..10 {
                        let n = SVector::<f64, 3>::new(theta[i].cos(), theta[i].sin(), 0.0);
                        let P = projection_matrix(n);
                        let eig = (P.transpose() * &tangent * &P).symmetric_eigenvalues();
                        let out = eig.min();
                        if out < min_eig {
                            min_eig = out;
                        }
                    }
                    output.set_scalar(Q::StabilityEigenvalue, ip, min_eig);
                }
            }
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
            let e_0 = input.get_scalar(Q::InternalElasticEnergy, ip);
            let del_elastic_strain = d_eps * del_t - self.model.del_plastic_strain();
            let e_1 = e_0 + 1. / density_mid * (sigma_mid.dot(&del_elastic_strain));
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
        HashMap::from([
            (Q::MandelStrainRate, QDim::Vector(6)),
            (Q::StabilityEigenvalue, QDim::Scalar),
        ])
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
