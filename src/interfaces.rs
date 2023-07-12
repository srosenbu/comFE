use nalgebra::{DVectorView, DVectorViewMut};
// A struct that contains an attribute for each possible quadrature value
// which is by default set to None.
#[derive(Debug)]
pub struct QValues<'a, T: 'a> {
    //_marker: marker::PhantomData<&'a T>,
    pub mandel_stress: Option<&'a T>,
    pub mandel_strain: Option<&'a T>,
    pub mandel_tangent: Option<&'a T>,
    pub nonlocal_strain: Option<&'a T>,
}

impl<'a, T> QValues<'a, T>  {
    pub fn set(&'a mut self, which: &str, value:&'a T ) {
        match which {
            "mandel_stress" => self.mandel_stress = Some(value),
            "mandel_strain" => self.mandel_strain = Some(value),
            "mandel_tangent" => self.mandel_tangent = Some(value),
            "nonlocal_strain" => self.nonlocal_strain = Some(value),
            _ => panic!("Invalid quadrature value"),
        }
    }
}

impl<'a, T> Default for QValues<'a, T> {
    fn default() -> Self {
        QValues {
            mandel_stress: None,
            mandel_strain: None,
            mandel_tangent: None,
            nonlocal_strain: None,
        }
    }
}

pub trait ConstitutiveModel {
    fn define_input(&self) -> QValues<usize>;

    fn define_output(&self) -> QValues<usize>;
    
    fn n_quadrature_points(&self, input: &QValues<DVectorView<f64>>, output: &mut QValues<DVectorViewMut<f64>>)-> Result<usize, &str>;
    
    fn evaluate_ip(
        &self,
        ip: usize,
        del_t: f64,
        input: &QValues<DVectorView<f64>>,
        output: &mut QValues<DVectorViewMut<f64>>,
    );
    fn evaluate(
        &self,
        del_t: f64,
        input: &QValues<DVectorView<f64>>,
        output: &mut QValues<DVectorViewMut<f64>>,
    ) {
        let n = self.n_quadrature_points(input, output).expect("Inconsistent number of quadrature points");
        for ip in 0..n {
            self.evaluate_ip(ip, del_t, input, output);
        }
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        input: &QValues<DVectorView<f64>>,
        output: &mut QValues<DVectorViewMut<f64>>,
        ips: &[usize],
    ) {
        for ip in ips {
            self.evaluate_ip(*ip, del_t, input, output);
        }
    }
}