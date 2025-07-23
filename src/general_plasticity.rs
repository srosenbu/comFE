use crate::QDim; // Ensure QDim is imported from the correct module
use crate::mandel::*;
use crate::{
    create_history_parameter_struct, impl_array_equivalent,
    interfaces::{ArrayEquivalent, ConstitutiveModelFn, StaticMap},
    q_dim_data_type,
};
use nalgebra::{SMatrix, SMatrixView, SVector, SVectorView};
pub trait Plasticity<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
>
{
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn new(parameters: &Self::Parameters) -> Self;
    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, STRESS_STRAIN>,
        sigma_1: &SVector<f64, STRESS_STRAIN>,
        del_eps: &SVector<f64, STRESS_STRAIN>,
        kappa: &SVector<f64, KAPPA>,
    );
    fn f(&self) -> f64;
    fn df_dsigma(&self) -> SVector<f64, STRESS_STRAIN>;
    fn df_dkappa(&self) -> SVector<f64, KAPPA>;
    fn g(&self) -> SVector<f64, STRESS_STRAIN>;
    fn dg_dkappa(&self) -> SMatrix<f64, STRESS_STRAIN, KAPPA>;
    fn dg_dsigma(&self) -> SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn k(&self) -> SVector<f64, KAPPA>;
    fn dk_dsigma(&self) -> SMatrix<f64, KAPPA, STRESS_STRAIN>;
    fn dk_dkappa(&self) -> SMatrix<f64, KAPPA, KAPPA>;
    fn elastic_tangent(&self) -> SMatrixView<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn elastic_tangent_inv(&self) -> SMatrixView<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn del_plastic_strain(&self) -> SVector<f64, STRESS_STRAIN>;
}
pub trait IsotropicPlasticity<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
>: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, 1>
{
}
struct IsotropicPlasticityModel3D<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> {
    model: MODEL,
}

create_history_parameter_struct!(
    IsotropicPlasticityHistory3D,
    2,
    7,
    [
        (alpha, (QDim::Scalar)),
        (plastic_strain, (QDim::RotatableVector(6)))
    ]
);

impl<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> ConstitutiveModelFn<6, 2, 7, N_PARAMETERS, PARAMETERS>
    for IsotropicPlasticityModel3D<N_PARAMETERS, PARAMETERS, MODEL>
{
    type History = IsotropicPlasticityHistory3D;
    type Parameters = MODEL::Parameters;
    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        _strain: &[f64; 6],
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; 7],
        parameters: &[f64; PARAMETERS],
    ) {
        let parameters_ = Self::Parameters::from_array(parameters);
        let history_ = IsotropicPlasticityHistory3D::from_array_mut(history);
        let mut model = MODEL::new(parameters_);

        let sigma_0 = SVectorView::<f64, 6>::from_array(stress).to_owned();
        let mut sigma_1 = SVector::<f64, 6>::zeros();
        let del_eps = SVectorView::<f64, 6>::from_array(del_strain).to_owned();
        let kappa = SVector::<f64, 1>::from_element(0.0); // Assuming kappa is a scalar

        model.set_model_state(&sigma_0, &sigma_1, &del_eps, &kappa);

        let f = model.f();
        if f <= 0.0 {
            *stress = sigma_1.into();
            if let Some(tangent) = tangent {
                *tangent = model.elastic_tangent().into();
            }
            return;
        }

        // Update stress and tangent
        *stress = (sigma_1 - model.del_plastic_strain()).into();
        if let Some(tangent) = tangent {
            *tangent = model.elastic_tangent().into();
        }
    }
}
