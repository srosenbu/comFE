use crate::general_plasticity::{IsotropicPlasticity, IsotropicPlasticityModel3D};
//#[cfg(feature = "python-bindings")]
use crate::interfaces::{ConstitutiveModelFn, evaluate_model};
use crate::mises_plasticity::MisesPlasticity3D;
use crate::linear_elasticity::LinearElasticity3D;
use crate::drucker_prager_classic::DruckerPragerClassic3D;

#[cfg(feature = "python-bindings")]
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
#[cfg(feature = "python-bindings")]
use pyo3::{pyclass, pymethods};

#[cfg(feature = "python-bindings")]
macro_rules! implement_python_model {
    ($name:ident, $model:ty, <$stress_strain:literal, $n_history:literal, $history:literal,$n_parameters:literal,$parameters:literal>) => {
        #[pyclass]
        pub struct $name(); 
        //{
        //    model: $model,
        //}
        #[pymethods]
        impl $name {
            fn evaluate(
                &self,
                time: f64,
                del_time: f64,
                strain: PyReadonlyArray1<f64>,
                del_strain: PyReadonlyArray1<f64>,
                mut stress: PyReadwriteArray1<f64>,
                mut tangent: Option<PyReadwriteArray1<f64>>,
                mut history: PyReadwriteArray1<f64>,
                parameters: PyReadonlyArray1<f64>,
            ) {
                let strain = strain.as_slice().unwrap();
                let del_strain = del_strain.as_slice().unwrap();
                let mut stress = stress.as_slice_mut().unwrap();
                let mut history = history.as_slice_mut().unwrap();
                let parameters = parameters.as_slice().unwrap();
                let tangent = tangent.as_mut();
                let tangent = match tangent {
                    Some(tangent) => Some(tangent.as_slice_mut().unwrap()),
                    None => None,
                };

                evaluate_model::<
                    $stress_strain,
                    $n_history,
                    $history,
                    $n_parameters,
                    $parameters,
                    $model,
                >(
                    time,
                    del_time,
                    strain,
                    del_strain,
                    &mut stress,
                    tangent,
                    &mut history,
                    parameters,
                );
            }
        }
    };
}

#[cfg(feature = "python-bindings")]
implement_python_model!(PyMisesPlasticity3D, MisesPlasticity3D, <6, 2, 7, 4, 4>);
#[cfg(feature = "python-bindings")]
implement_python_model!(PyLinearElasticity3D, LinearElasticity3D, <6, 0, 0, 2, 2>);
#[cfg(feature = "python-bindings")]
implement_python_model!(PyDruckerPragerClassic3D, IsotropicPlasticityModel3D<4,4,DruckerPragerClassic3D>, <6, 2, 7, 4, 4>);
