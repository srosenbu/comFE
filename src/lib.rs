use std::collections::HashMap;

use nalgebra::{Const, DVectorView, DVectorViewMut, Dyn, SMatrix};
use numpy::{PyArray2, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::smallstrain::SmallStrainModel;
use crate::smallstrain::linear_elastic::LinearElastic3D;
use crate::interfaces::{QValues,ConstitutiveModel};
pub mod smallstrain;
pub mod interfaces;

#[pyclass(unsendable)]
struct PyConstitutiveModel {
    model: Box<dyn ConstitutiveModel>,
}



/// Wrapper struct for LinearElastic3D in python
#[pyclass(unsendable)]
struct PySmallStrainModel {
    model: Box<dyn SmallStrainModel>,
}

#[pymethods]
impl PySmallStrainModel {
    fn evaluate(
        &self,
        del_t: f64,
        stress: PyReadwriteArray1<f64>,
        del_strain: PyReadonlyArray1<f64>,
        tangent: PyReadwriteArray1<f64>,
    ) -> PyResult<()> {
        let mut stress = stress
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let mut tangent = tangent
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let del_strain = del_strain
            .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        self.model
            .evaluate(del_t, &mut stress, &del_strain, &mut tangent);
        Ok(())
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        stress: PyReadwriteArray1<f64>,
        del_strain: PyReadonlyArray1<f64>,
        tangent: PyReadwriteArray1<f64>,
        ips: PyReadonlyArray1<usize>,
    ) -> PyResult<()> {
        let mut stress = stress
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let mut tangent = tangent
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let del_strain = del_strain
            .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let ips = ips.as_slice().unwrap();
        self.model.evaluate_some(
            del_t,
            &mut stress,
            &del_strain,
            &mut tangent,
            ips,
        );
        Ok(())
    }
}

// #[pyfunction]
// fn py_new_linear_elastic_3d(parameters: &PyDict) -> PyResult<PySmallStrainModel> {
//     let parameters = parameters.extract::<HashMap<String, f64>>().unwrap();
//     let model = LinearElastic3D::new(&parameters);
//     Ok(PySmallStrainModel {
//         model: Box::new(model),
//     })
// }
#[pyfunction]
fn py_new_linear_elastic_3d(parameters: &PyDict) -> PyResult<PyConstitutiveModel> {
    let parameters = parameters.extract::<HashMap<String, f64>>().unwrap();
    let model = LinearElastic3D::new(&parameters);
    Ok(PyConstitutiveModel {
        model: Box::new(model),
    })
}
#[pyclass]
struct PyLinearElastic3D {
    model: LinearElastic3D,
}

#[pymethods]
impl PyLinearElastic3D {
    #[new]
    fn new(parameters: &PyDict) -> Self {
        let mut D = SMatrix::<f64, 6, 6>::zeros();
        let E = parameters.get_item("E").unwrap().extract::<f64>().unwrap();
        let nu = parameters.get_item("nu").unwrap().extract::<f64>().unwrap();
        let c1 = E / (1.0 + nu) / (1.0 - 2.0 * nu);
        let c2 = c1 * (1.0 - nu);
        D[(0, 0)] = c2;
        D[(1, 1)] = c2;
        D[(2, 2)] = c2;
        D[(0, 1)] = E * nu / (1.0 - nu);
        D[(0, 2)] = E * nu / (1.0 - nu);
        D[(1, 0)] = E * nu / (1.0 - nu);
        D[(1, 2)] = E * nu / (1.0 - nu);
        D[(2, 0)] = E * nu / (1.0 - nu);
        D[(2, 1)] = E * nu / (1.0 - nu);
        D[(3, 3)] = c1;
        D[(4, 4)] = c1;
        D[(5, 5)] = c1;
        Self {
            model: LinearElastic3D { D: D },
        }
    }

    fn evaluate(
        &self,
        del_t: f64,
        stress: PyReadwriteArray1<f64>,
        del_strain: PyReadonlyArray1<f64>,
        tangent: PyReadwriteArray1<f64>,
    ) -> PyResult<()> {
        let mut stress = stress
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let mut tangent = tangent
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let del_strain = del_strain
            .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        self.model
            .evaluate(del_t, &mut stress, &del_strain, &mut tangent);
        Ok(())
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        stress: PyReadwriteArray1<f64>,
        del_strain: PyReadonlyArray1<f64>,
        tangent: PyReadwriteArray1<f64>,
        ips: PyReadonlyArray1<usize>,
    ) -> PyResult<()> {
        let mut stress = stress
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let mut tangent = tangent
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let del_strain = del_strain
            .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let ips = ips.as_slice().unwrap();
        self.model.evaluate_some(
            del_t,
            &mut stress,
            &del_strain,
            &mut tangent,
            ips,
        );
        Ok(())
    }
}
#[pyfunction]
fn dict_input(input: HashMap<String,PyReadwriteArray1<f64>>) -> PyResult<()>{
    let mut strain = input.get("strain").expect("Strain not defined").try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
    strain[0] = 42.;
    Ok(())
}

// #[pyfunction]
// fn q_values_from_py(input: HashMap<String, PyReadonlyArray1<f64>>) -> PyResult<()> {
//     let strain = input.get("strain").expect("Strain not defined").try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
//     let stress = input.get("stress").expect("Stress not defined").try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
//     let tangent = input.get("tangent").expect("Tangent not defined").try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
//     let q_values = QValues::<DVectorView::<f64>> {
//         mandel_strain: Some(&strain),
//         mandel_stress: Some(&stress),
//         mandel_tangent: Some(&tangent),
//     };
//     println!("{:?}", q_values);

//     Ok(())
// }

#[pyfunction]
fn q_values_mut_from_py(input: HashMap<String, PyReadwriteArray1<f64>>) -> PyResult<()> {
    let strain = input.get("strain").expect("Strain not defined").try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
    let stress = input.get("stress").expect("Stress not defined").try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
    let tangent = input.get("tangent").expect("Tangent not defined").try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
    let q_values = QValues::<DVectorViewMut::<f64>> {
        mandel_strain: Some(&strain),
        mandel_stress: Some(&stress),
        mandel_tangent: Some(&tangent),
        ..Default::default()
    };
    
    println!("{:?}", &q_values);

    Ok(())
}
/// A Python module implemented in Rust.
#[pymodule]
fn comfe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLinearElastic3D>()?;
    m.add_class::<PySmallStrainModel>()?;
    m.add_function(wrap_pyfunction!(py_new_linear_elastic_3d, m)?)?;
    m.add_function(wrap_pyfunction!(dict_input,m)?)?;
    //m.add_function(wrap_pyfunction!(q_values_from_py,m)?)?;
    m.add_function(wrap_pyfunction!(q_values_mut_from_py,m)?)?;
    Ok(())
}
