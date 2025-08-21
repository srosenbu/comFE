use comfe::bindings::*;
use comfe::implement_python_model;
use comfe::linear_elasticity::LinearElasticity3D;
use comfe::mises_plasticity::MisesPlasticity3D;
use comfe::plasticity::{DruckerPrager3D, IsotropicPlasticityModel3D, DruckerPragerHyperbolic3D};

use pyo3::prelude::*;

#[pyfunction]
fn double(x: usize) -> usize {
    x * 2
}


#[pymodule]
fn comfe_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)
}