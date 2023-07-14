use std::collections::HashMap;
//use strum::IntoEnumIterator;
use strum_macros::{EnumIter, EnumString, ToString};

use nalgebra::{Const, DVectorView, DVectorViewMut, Dyn, Matrix, SVector, ViewStorage};
// A struct that contains an attribute for each possible quadrature value
// which is by default set to None.
#[derive(Debug)]
pub struct QValues<'a, T: 'a> {
    //_marker: marker::PhantomData<&'a T>,
    pub mandel_stress: &'a Option<T>,
    pub mandel_strain: &'a Option<T>,
    pub mandel_tangent: &'a Option<T>,
    pub nonlocal_strain: &'a Option<T>,
}
#[derive(Debug, EnumIter, Hash, PartialEq, Eq, EnumString, ToString)]
pub enum Q {
    #[strum(serialize = "MandelStress", serialize = "mandel_stress")]
    MandelStress,
    #[strum(serialize = "MandelStrain", serialize = "mandel_strain")]
    MandelStrain,
    #[strum(serialize = "MandelTangent", serialize = "mandel_tangent")]
    MandelTangent,
    #[strum(serialize = "NonlocalStrain", serialize = "nonlocal_strain")]
    NonlocalStrain,
    #[strum(serialize = "LAST", serialize = "last")]
    LAST,
}
#[derive(Debug)]
pub struct QValueInput<'a> {
    pub data: [Option<DVectorView<'a, f64>>; Q::LAST as usize],
}
#[derive(Debug)]
pub struct QValueOutput<'a> {
    pub data: [Option<DVectorViewMut<'a, f64>>; Q::LAST as usize],
}
impl QValueInput<'_> {
    pub fn get_data(&self, q: Q) -> &DVectorView<f64> {
        self.data[q as usize].as_ref().unwrap()
    }
    pub fn get_scalar(&self, q: Q, i: usize) -> f64 {
        self.data[q as usize].unwrap()[(i)]
    }
    pub fn get_vector<const SIZE: usize>(
        &self,
        q: Q,
        i: usize,
    ) -> Matrix<
        f64,
        Const<SIZE>,
        Const<1>,
        ViewStorage<'_, f64, Const<SIZE>, Const<1>, Const<1>, Dyn>,
    > {
        self.data[q as usize]
            .as_ref()
            .unwrap()
            .fixed_view::<SIZE, 1>(i * SIZE, 0)
    }
    pub fn get_slice<const SIZE: usize>(&self, q: Q, i: usize) -> &[f64] {
        self.data[q as usize].as_ref().unwrap().as_slice()[i * SIZE..(i + 1) * SIZE].as_ref()
    }
}
impl QValueOutput<'_> {
    pub fn get_data(&mut self, q: Q) -> &DVectorViewMut<f64> {
        self.data[q as usize].as_mut().unwrap()
    }
    pub fn set_scalar(&mut self, q: Q, i: usize, value: f64) {
        self.data[q as usize].as_mut().unwrap()[(i)] = value;
    }
    pub fn set_vector<const SIZE: usize>(&mut self, q: Q, i: usize, value: SVector<f64, SIZE>) {
        self.data[q as usize]
            .as_mut()
            .unwrap()
            .fixed_view_mut::<SIZE, 1>(i * SIZE, 0)
            .copy_from(&value);
    }
    pub fn set_slice<const SIZE: usize>(&mut self, q: Q, i: usize, value: &[f64]) {
        self.data[q as usize]
            .as_mut()
            .unwrap()
            .fixed_view_mut::<SIZE, 1>(i * SIZE, 0)
            .copy_from_slice(value);
    }
}

pub enum QDim {
    Scalar,
    Vector(usize),
    Tensor(usize, usize),
}

impl QDim {
    pub fn size(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(n) => *n,
            QDim::Tensor(m, n) => *m * *n,
        }
    }
}

impl<'a, T> Default for QValues<'a, T> {
    fn default() -> Self {
        QValues {
            mandel_stress: &None,
            mandel_strain: &None,
            mandel_tangent: &None,
            nonlocal_strain: &None,
        }
    }
}

pub trait ConstitutiveModel {
    //Mainly for the purpose of telling Python what is needed
    fn define_input(&self) -> HashMap<Q, QDim>;
    //Mainly for the purpose of telling Python what is needed
    fn define_output(&self) -> HashMap<Q, QDim>;

    fn evaluate_ip(&self, ip: usize, del_t: f64, input: &QValueInput, output: &mut QValueOutput);
    fn evaluate(&self, del_t: f64, input: &QValueInput, output: &mut QValueOutput) {
        let n = self.check_size_and_return(input, output).expect("");
        for i in 0..n {
            self.evaluate_ip(i, del_t, input, output);
        }
    }
    fn evaluate_some(
        &self,
        del_t: f64,
        input: &QValueInput,
        output: &mut QValueOutput,
        ip_list: &[usize],
    ) {
        let _ = self.check_size_and_return(input, output).expect("");
        for i in ip_list {
            self.evaluate_ip(*i, del_t, input, output);
        }
    }
    fn check_size_and_return(
        &self,
        input: &QValueInput,
        output: &mut QValueOutput,
    ) -> Result<usize, &str> {
        let input_def = self.define_input();
        let output_def = self.define_output();
        let mut sizes = Vec::<usize>::new();
        for (q, dim) in input_def {
            sizes.push(input.get_data(q).len() / dim.size());
        }
        for (q, dim) in output_def {
            sizes.push(output.get_data(q).len() / dim.size());
        }
        sizes.sort();
        if sizes[0] == sizes[sizes.len() - 1] {
            return Ok(sizes[0]);
        } else {
            return Err("There are inconsistencies in input and output sizes.");
        }
    }
}
