use std::collections::HashMap;

use crate::interfaces::{ConstitutiveModel, QDim, QValueInput, QValueOutput, Q};
use crate::jh2::JH23D;
use crate::smallstrain::linear_elastic::LinearElastic3D;
//use crate::stress_strain;
use nalgebra::{Const, DVectorView, DVectorViewMut, Dyn, SMatrix};
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::str::FromStr;
pub mod interfaces;
pub mod smallstrain;
pub mod jh2;
pub mod stress_strain;

#[pyclass(unsendable)]
struct PyConstitutiveModel {
    model: Box<dyn ConstitutiveModel>,
}

#[pymethods]
impl PyConstitutiveModel {
    fn evaluate(
        &self,
        del_t: f64,
        input: HashMap<String, PyReadonlyArray1<f64>>,
        output: HashMap<String, PyReadwriteArray1<f64>>,
    ) -> PyResult<()> {
        let mut input_data: [Option<DVectorView<f64>>; Q::LAST as usize] =
            std::array::from_fn(|_| None);

        let mut output_data: [Option<DVectorViewMut<f64>>; Q::LAST as usize] =
            std::array::from_fn(|_| None);
        //let mut q_output = HashMap::<Q, DVectorViewMut<f64>>::new();

        for (key, value) in input.iter() {
            let q = Q::from_str(key).expect(&format!("Failed to read dictionary from Python. Name {} unknown", key));
            input_data[q as usize] = Some(
                value
                    .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
                    .unwrap(),
            );
        }
        for (key, value) in output.iter() {
            let q = Q::from_str(key).expect(&format!("Failed to read dictionary from Python. Name {} unknown", key));
            output_data[q as usize] = Some(
                value
                    .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
                    .unwrap(),
            );
        }
        let q_input = QValueInput::new(input_data);
        let mut q_output = QValueOutput::new(output_data);
        self.model.evaluate(del_t, &q_input, &mut q_output);
        Ok(())
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        input: HashMap<String, PyReadonlyArray1<f64>>,
        output: HashMap<String, PyReadwriteArray1<f64>>,
        ips: PyReadonlyArray1<usize>,
    ) -> PyResult<()> {
        let mut input_data: [Option<DVectorView<f64>>; Q::LAST as usize] =
            std::array::from_fn(|_| None);

        let mut output_data: [Option<DVectorViewMut<f64>>; Q::LAST as usize] =
            std::array::from_fn(|_| None);


        for (key, value) in input.iter() {
            let q = Q::from_str(key).expect(&format!("Failed to read dictionary from Python. Name {} unknown", key));
            input_data[q as usize] = Some(
                value
                    .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
                    .unwrap(),
            );
        }
        for (key, value) in output.iter() {
            let q = Q::from_str(key).expect(&format!("Failed to read dictionary from Python. Name {} unknown", key));
            output_data[q as usize] = Some(
                value
                    .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
                    .unwrap(),
            );
        }

        let q_input = QValueInput::new(input_data);
        let mut q_output = QValueOutput::new(output_data);

        self.model
            .evaluate_some(del_t, &q_input, &mut q_output, ips.as_slice().unwrap());
        Ok(())
    }
    fn define_input(&self, py: Python) -> PyResult<PyObject> {
        let input_py = PyDict::new(py);
        let input_rs = self.model.define_input();
        for (key, value) in input_rs.iter() {
            match value {
                QDim::Scalar => {
                    input_py.set_item(key.to_string(), 1)?;
                }
                QDim::Vector(n) => {
                    input_py.set_item(key.to_string(), n)?;
                }
                QDim::Tensor(n, m) => {
                    input_py.set_item(key.to_string(), (n, m))?;
                }
            }
        }
        Ok(input_py.into())
    }
    fn define_output(&self, py: Python) -> PyResult<PyObject> {
        let output_py = PyDict::new(py);
        let output_rs = self.model.define_output();
        for (key, value) in output_rs.iter() {
            match value {
                QDim::Scalar => {
                    output_py.set_item(key.to_string(), 1)?;
                }
                QDim::Vector(n) => {
                    output_py.set_item(key.to_string(), n)?;
                }
                QDim::Tensor(n, m) => {
                    output_py.set_item(key.to_string(), (n, m))?;
                }
            }
        }
        Ok(output_py.into())
    }
}

#[pyclass]
struct PyLinearElastic3D {
    D: SMatrix<f64, 6, 6>,
}
impl PyLinearElastic3D {
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
#[pymethods]
impl PyLinearElastic3D {
    #[new]
    fn new(parameters: &PyDict) -> Self {
        let mut D = SMatrix::<f64, 6, 6>::zeros();
        let E = parameters.get_item("E").unwrap().extract::<f64>().expect("Failed to read youngs modulus from Python");
        let nu = parameters.get_item("nu").unwrap().extract::<f64>().expect("Failed to read poisson ratio from Python");
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
        Self { D: D }
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

        for ip in 0..stress.nrows() / 6 {
            self.evaluate_ip(ip, del_t, &mut stress, &del_strain, &mut tangent)
        }

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
        for ip in ips {
            self.evaluate_ip(*ip, del_t, &mut stress, &del_strain, &mut tangent)
        }
        Ok(())
    }
}

#[pyfunction]
fn py_new_linear_elastic_3d(parameters: HashMap<String, f64>) -> PyResult<PyConstitutiveModel> {
    let linear_elastic = LinearElastic3D::new(&parameters);
    Ok(PyConstitutiveModel {
        model: Box::new(linear_elastic),
    })
}

#[pymodule]
fn comfe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLinearElastic3D>()?;
    m.add_function(wrap_pyfunction!(py_new_linear_elastic_3d, m)?)?;
    Ok(())
}
