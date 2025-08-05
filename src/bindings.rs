use crate::general_plasticity::{IsotropicPlasticity, IsotropicPlasticityModel3D};
//#[cfg(feature = "python-bindings")]
use crate::interfaces::{ConstitutiveModelFn, evaluate_model};
use crate::mises_plasticity::MisesPlasticity3D;
use crate::linear_elasticity::LinearElasticity3D;
use crate::drucker_prager_classic::DruckerPragerClassic3D;
use std::collections::HashMap;
#[cfg(feature = "python-bindings")]
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
#[cfg(feature = "python-bindings")]
use pyo3::{pyclass, pymethods};

#[cfg(feature = "python-bindings")]
#[pyclass]
enum StressStrainConstraint {
    UNIAXIAL_STRAIN = 1,
    UNIAXIAL_STRESS = 2,
    PLANE_STRAIN = 3,
    PLANE_STRESS = 4,
    FULL = 5,
}
#[cfg(feature = "python-bindings")]
macro_rules! implement_python_model {
    ($name:ident, $model:ty) => {
        #[pyclass]
        pub struct $name(); 

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
                <$model>::evaluate_all(
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
            fn history_dim(&self) -> HashMap<&str, usize> {
                HashMap::from([("history",<$model>::HISTORY)])
            }
            fn stress_strain_dim(&self) -> usize {
                <$model>::STRESS_STRAIN
            }
            fn geometry_dim(&self) -> usize {
                match <$model>::STRESS_STRAIN {
                    6 => 3,
                    4 => 2,
                    1 => 1,
                    _ => panic!("Unsupported stress-strain dimension"),
                }
            }
        }
    };
}

#[cfg(feature = "python-bindings")]
implement_python_model!(PyMisesPlasticity3D, MisesPlasticity3D);
#[cfg(feature = "python-bindings")]
implement_python_model!(PyLinearElasticity3D, LinearElasticity3D);
#[cfg(feature = "python-bindings")]
implement_python_model!(PyDruckerPragerClassic3D, IsotropicPlasticityModel3D<4,4,DruckerPragerClassic3D>);
