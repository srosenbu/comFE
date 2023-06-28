use pyo3::prelude::*;
use nalgebra::{SMatrix, Dyn, Const, DVectorViewMut, DVectorView};
use numpy::{PyReadwriteArray1, PyReadonlyArray1, PyArray2};

struct LinearElastic3D{
    D: SMatrix<f64, 6, 6>,
}


trait SmallStrainModel{
    fn evaluate_ip(&self, ip: usize, del_t:f64, stress: &mut DVectorViewMut<f64>, del_strain: &DVectorView<f64>);
    fn evaluate(&self, del_t:f64, stress: &mut DVectorViewMut<f64>, del_strain: &DVectorView<f64>){
        assert_eq!(stress.nrows(), del_strain.nrows());
        let n: usize = stress.nrows()/6;
        assert_eq!(stress.nrows(), n*6);
        for ip in 0..n{
            println!("ip: {}", ip);
            self.evaluate_ip(ip, del_t, stress, del_strain);
        }
    }
}

/// implement the small strain model for linear elastic material
impl SmallStrainModel for LinearElastic3D{
    fn evaluate_ip(&self, ip: usize, del_t:f64, stress: &mut DVectorViewMut<f64>, del_strain: &DVectorView<f64>){
        println!("shape: {} {}", stress.nrows(), stress.ncols());
        let mut view_stress = stress.fixed_view_mut::<6, 1>(ip*6,0);
        let view_strain = del_strain.fixed_view::<6, 1>(ip*6,0);
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
    #[new]
    fn new(E: f64, nu: f64)->Self{
        println!("E: {}, nu: {}", E, nu);
        let mut D = SMatrix::<f64, 6, 6>::zeros();
        let c1 = E/(1.0+nu)/(1.0-2.0*nu);
        let c2 = c1*(1.0-nu);
        D[(0,0)] = c2;
        D[(1,1)] = c2;
        D[(2,2)] = c2;
        D[(0,1)] = E*nu/(1.0-nu);
        D[(0,2)] = E*nu/(1.0-nu);
        D[(1,0)] = E*nu/(1.0-nu);
        D[(1,2)] = E*nu/(1.0-nu);
        D[(2,0)] = E*nu/(1.0-nu);
        D[(2,1)] = E*nu/(1.0-nu);
        D[(3,3)] = c1;
        D[(4,4)] = c1;
        D[(5,5)] = c1;
        Self{
            model: LinearElastic3D{
                D: D,
            }
        }
    }

    fn evaluate(&self, del_t:f64, stress: PyReadwriteArray1<f64>, del_strain: PyReadonlyArray1<f64>)->PyResult<()>{
        let mut stress = stress.try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
        let del_strain = del_strain.try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>().unwrap();
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
    m.add_class::<PyLinearElastic3D>()?;
    Ok(())
}