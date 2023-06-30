use std::collections::HashMap;

use nalgebra::{Const, DVectorView, DVectorViewMut, Dyn, SMatrix};
use numpy::{PyArray2, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct LinearElastic3D {
    D: SMatrix<f64, 6, 6>,
}

trait SmallStrainModel {
    //fn new(parameters: &HashMap<String, f64>) -> Self;
    fn evaluate_ip(
        &self,
        ip: usize,
        del_t: f64,
        stress: &mut DVectorViewMut<f64>,
        del_strain: &DVectorView<f64>,
        tangent: &mut DVectorViewMut<f64>,
    );
    fn evaluate(
        &self,
        del_t: f64,
        stress: &mut DVectorViewMut<f64>,
        del_strain: &DVectorView<f64>,
        tangent: &mut DVectorViewMut<f64>,
    ) {
        assert_eq!(stress.nrows(), del_strain.nrows());
        let n: usize = stress.nrows() / 6;
        assert_eq!(stress.nrows(), n * 6);
        for ip in 0..n {
            self.evaluate_ip(ip, del_t, stress, del_strain, tangent);
        }
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        stress: &mut DVectorViewMut<f64>,
        del_strain: &DVectorView<f64>,
        tangent: &mut DVectorViewMut<f64>,
        ips: &[usize],
    ) {
        assert_eq!(stress.nrows(), del_strain.nrows());
        let n: usize = stress.nrows() / 6;
        assert_eq!(stress.nrows(), n * 6);
        for ip in ips {
            self.evaluate_ip(*ip, del_t, stress, del_strain, tangent);
        }
    }
}

/// Wrapper struct for LinearElastic3D in python
#[pyclass(unsendable)]
struct PySmallStrainModel {
    model: Box<dyn SmallStrainModel>,
}
/// implement the small strain model for linear elastic material
impl SmallStrainModel for LinearElastic3D {

    fn evaluate_ip(
        &self,
        ip: usize,
        del_t: f64,
        stress: &mut DVectorViewMut<f64>,
        del_strain: &DVectorView<f64>,
        tangent: &mut DVectorViewMut<f64>,
    ) {
        let mut view_stress = stress.fixed_view_mut::<6, 1>(ip * 6, 0);
        let mut view_tangent = tangent.fixed_view_mut::<36, 1>(ip * 36, 0);
        let view_strain = del_strain.fixed_view::<6, 1>(ip * 6, 0);
        view_stress += self.D * view_strain;
        view_tangent.copy_from_slice(&self.D.as_slice());
    }
}
impl LinearElastic3D {
    fn new(parameters: &HashMap<String, f64>) -> Self{
        //let mut D = SMatrix::<f64, 6, 6>::zeros();
        let E = parameters.get("E").unwrap();
        let nu = parameters.get("nu").unwrap();
        let mu = E / (2.0 * (1.0 + nu));
        let lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let D = SMatrix::<f64, 6, 6>::new(
            lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0,
            lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0,
            lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0,
            0.0,0.0,0.0,0.0,0.0, 2.0 * mu,
        );
        Self { D: D }
    }
}
impl PySmallStrainModel {
    fn new(parameters: &PyDict) -> Self{
        let mut parameters_map = HashMap::new();
        for (key, value) in parameters.iter() {
            let key = key.extract::<String>().unwrap();
            let value = value.extract::<f64>().unwrap();
            parameters_map.insert(key, value);
        }
        Self { model: ::new(&parameters_map) }
    }
    fn evaluate(
        &self,
        del_t: f64,
        stress: &mut PyReadwriteArray1<f64>,
        del_strain: &PyReadonlyArray1<f64>,
        tangent: &mut PyReadwriteArray1<f64>,
    ) {
        assert_eq!(stress.len(), del_strain.len());
        let n: usize = stress.len() / 6;
        assert_eq!(stress.len(), n * 6);
        for ip in 0..n {
            self.model.evaluate_ip(ip, del_t, stress, del_strain, tangent);
        }
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        stress: &mut PyReadwriteArray1<f64>,
        del_strain: &PyReadonlyArray1<f64>,
        tangent: &mut PyReadwriteArray1<f64>,
        ips: &[usize],
    ) {
        assert_eq!(stress.len(), del_strain.len());
        let n: usize = stress.len() / 6;
        assert_eq!(stress.len(), n * 6);
        for ip in ips {
            self.model.evaluate_ip(*ip, del_t, stress, del_strain, tangent);
        }
    }
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
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn comfe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<PyLinearElastic3D>()?;
    Ok(())
}
