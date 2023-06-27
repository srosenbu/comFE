use pyo3::prelude::*;
use numpy::nalgebra::{SMatrix, DVector};
use numpy::{PyReadwriteArray1, PyReadonlyArray1};

struct LinearElastic3D{
    D: SMatrix<f64, 6, 6>,
    //n: usize,
    //ips: Option<Vec<usize>>,
}


trait SmallStrainModel{
    fn evaluate_ip(&self, ip: usize, del_t:f64, stress: &mut DVector<f64>, del_strain: &DVector<f64>);
    fn evaluate(&self, del_t:f64, stress: &mut DVector<f64>, del_strain: &DVector<f64>){
        assert_eq!(stress.nrows(), del_strain.nrows());
        // match ips{
        //     Some(ips) => {
        //         for ip in ips{
        //             self.evaluate_ip(ip, del_t, stress, del_strain);
        //         }
        //     },
        //     None => {
                let n: usize = stress.nrows()/6;
                assert_eq!(stress.nrows(), n*6);
                for ip in 0..n{
                    self.evaluate_ip(ip, del_t, stress, del_strain);
                }
        //    }
        //}
    }
}

/// implement the small strain model for linear elastic material
impl SmallStrainModel for LinearElastic3D{
    fn evaluate_ip(&self, ip: usize, del_t:f64, stress: &mut DVector<f64>, del_strain: &DVector<f64>){
        let mut view_stress = stress.fixed_view_mut::<6, 1>(ip*6,1);
        let view_strain = del_strain.fixed_view::<6, 1>(ip*6,1);
        //view_stress.copy_from(&(self.D * view_strain));
        view_stress += self.D * view_strain;
    }
}
/// Wrapper struct for LinearElastic3D in python
#[pyclass]
struct PyLinearElastic3D{
    model: LinearElastic3D,
}
trait PySmallStrainModel{
    fn evaluate(&self, del_t:f64, stress: PyReadwriteArray1<f64>, del_strain: PyReadonlyArray1<f64>);
}

#[pymethods] 
impl  PyLinearElastic3D{
    fn evaluate(&self, del_t:f64, stress: PyReadwriteArray1<f64>, del_strain: PyReadonlyArray1<f64>)->PyResult<()>{
        let mut stress = stress.as_matrix_mut();
        let del_strain = del_strain.as_matrix();
        self.model.evaluate(del_t, &mut stress, &del_strain);
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
    Ok(())
}