use crate::general_plasticity::*;
use crate::consts::*;
use crate::impl_array_equivalent;
use crate::interfaces::*;
use crate::mandel::*;
use crate::{create_history_parameter_struct, q_dim_data_type};
//use crate::impl_from_array;
use nalgebra::{SMatrix, SMatrixViewMut, SVector, SVectorView, SVectorViewMut};


create_history_parameter_struct!(
    DruckerPragerClassicParameters,
    4,
    4,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (a, (QDim::Scalar)),
        (b, (QDim::Scalar))
    ]
);
#[derive(Default, Clone, Copy)]
pub struct DruckerPragerClassic3D{
    elastic_tangent: SMatrix<f64, 6, 6>,
    elastic_tangent_inv: SMatrix<f64, 6, 6>,
    f: f64,
    df_dsigma: SVector<f64, 6>,
    df_dkappa: SVector<f64, 1>,
    g: SVector<f64, 6>,
    dg_dkappa: SMatrix<f64, 6, 1>,
    dg_dsigma: SMatrix<f64, 6, 6>,
    k: SVector<f64, 1>,
    dk_dsigma: SMatrix<f64, 1, 6>,
    dk_dkappa: SMatrix<f64, 1, 1>,
    del_plastic_strain: SVector<f64, 6>,
}

impl Plasticity<6, 4,4, 1> for DruckerPragerClassic3D {
    type Parameters = DruckerPragerClassicParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        let elastic_tangent = (2.0*parameters.mu) * const { projection_dev::<6>() } + parameters.kappa * const { sym_id_outer_sym_id::<6>() };
        let elastic_tangent_inv = elastic_tangent.try_inverse().expect("D must be invertible");
        
        DruckerPragerClassic3D{
            elastic_tangent,
            elastic_tangent_inv,
            ..Default::default()
        }
    }

    fn set_model_state(
        &mut self,
        _sigma_0: &SVector<f64, 6>,
        _sigma_1: &SVector<f64, 6>,
        _del_eps: &SVector<f64, 6>,
        _kappa: &SVector<f64, 1>,
    ) {
        // Implementation of setting model state
    }

    fn f(&self) -> f64 {
        // Implementation of f function
        self.f
    }
    fn df_dsigma(&self) -> &SVector<f64, 6> {
        // Implementation of df_dsigma
        &self.df_dsigma
    }
    fn df_dkappa(&self) -> &SVector<f64, 1> {
        // Implementation of df_dkappa
        &self.df_dkappa
    }
    fn g(&self) -> &SVector<f64, 6> {
        // Implementation of g function
        &self.g
    }
    fn dg_dkappa(&self) -> &SMatrix<f64, 6, 1> {
        // Implementation of dg_dkappa
        &self.dg_dkappa
    }
    fn dg_dsigma(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of dg_dsigma
        &self.dg_dsigma
    }
    fn k(&self) -> &SVector<f64, 1> {
        // Implementation of k function
        &self.k
    }
    fn dk_dsigma(&self) -> &SMatrix<f64, 1, 6> {
        // Implementation of dk_dsigma
        &self.dk_dsigma
    }
    fn dk_dkappa(&self) -> &SMatrix<f64, 1, 1> {
        // Implementation of dk_dkappa
        &self.dk_dkappa
    }
    fn elastic_tangent(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent
        &self.elastic_tangent
    }
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent_inv
        &self.elastic_tangent_inv
    }
    fn del_plastic_strain(&self) -> &SVector<f64, 6> {
        // Implementation of del_plastic_strain
        &self.del_plastic_strain
    }

}