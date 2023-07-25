use core::panic;
use std::collections::HashMap;
//use strum::IntoEnumIterator;
use strum_macros::{EnumIter, EnumString, ToString};

use nalgebra::{Const, DVectorView, DVectorViewMut, Dyn, Matrix, SVector, ViewStorage, SMatrix};

#[derive(Debug, EnumIter, Hash, PartialEq, Eq, EnumString, ToString)]
pub enum Q {
    #[strum(serialize = "MandelStress", serialize = "mandel_stress")]
    MandelStress,
    #[strum(serialize = "MandelStrain", serialize = "mandel_strain")]
    MandelStrain,
    #[strum(serialize = "MandelStrainRate", serialize = "mandel_strain_rate")]
    MandelStrainRate,
    #[strum(serialize = "MandelTangent", serialize = "mandel_tangent")]
    MandelTangent,
    #[strum(serialize = "VelocityGradient", serialize = "velocity_gradient")]
    VelocityGradient,
    #[strum(serialize = "NonlocalEquivalentStrain", serialize = "nonlocal_equivalent_strain")]
    NonlocalEquivalentStrain,
    #[strum(serialize = "Lambda", serialize = "lambda")]
    Lambda,
    #[strum(serialize =  "Density", serialize = "density")]
    Density,
    #[strum(serialize = "Pressure", serialize = "pressure")]
    Pressure,
    #[strum(serialize = "Damage", serialize = "damage")]
    Damage,
    #[strum(serialize = "StrainRateNorm", serialize = "strain_rate_norm")]
    StrainRateNorm,
    #[strum(serialize = "EquivalentPlasticStrain", serialize = "equivalent_plastic_strain")]
    EquivalentPlasticStrain,
    #[strum(serialize = "MisesStress", serialize = "mises_stress")]
    MisesStress,
    #[strum(serialize = "LAST", serialize = "last")]
    LAST,
}

#[derive(Debug)]
pub struct QValueInput<'a> {
    data: [Option<DVectorView<'a, f64>>; Q::LAST as usize],
}

#[derive(Debug)]
pub struct QValueOutput<'a> {
    data: [Option<DVectorViewMut<'a, f64>>; Q::LAST as usize],
}
impl<'a> QValueInput<'a> {
    //TODO: When the const generics are stabilized, most of this should be changed to sth like
    // ```rust
    // pub fn get<Q>(&self) -> SMatrix<f64, Q::n, Q::m> { ... }; 
    // pub fn get<Q>(&self) -> Q::AssociatedType){ ... }; 
    // ```

    pub fn new<'b: 'a>(data: [Option<DVectorView<'b, f64>>; Q::LAST as usize]) -> Self{
        Self { data }
    }
    pub fn get_data(&self, q: Q) -> &DVectorView<f64> {
        self.data[q as usize].as_ref().unwrap()
    }
    pub fn get_scalar(&self, q: Q, i: usize) -> f64 {
        self.data[q as usize].unwrap()[i]
    }
    pub fn get_vector_view<const SIZE: usize>(
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
    pub fn get_vector<const SIZE: usize>(
        &self,
        q: Q,
        i: usize,
    ) -> SVector<f64, SIZE>
    {
        SVector::<f64, SIZE>::from_column_slice(self.get_slice::<SIZE>(q, i))
    }
    pub fn get_tensor<const DIM: usize, const SIZE: usize>(&self, q:Q, i:usize) -> SMatrix<f64, DIM,DIM> {
        SMatrix::<f64, DIM, DIM>::from_row_slice(self.get_slice::<{SIZE}>(q, i))
    }
    pub fn get_slice<const SIZE: usize>(&self, q: Q, i: usize) -> &[f64] {
        self.data[q as usize].as_ref().unwrap().as_slice()[i * SIZE..(i + 1) * SIZE].as_ref()
    }
}
impl<'a> QValueOutput<'a> {
    //TODO: When the const generics are stabilized, most of this should be changed to sth like
    // ```rust
    // pub fn set<Q>(&self, value: SMatrix<f64, Q::n, Q::m>){ ... }; 
    // pub fn set<Q>(&self, value: Q::AssociatedType){ ... }; 
    // ```

    pub fn new<'b: 'a>(data: [Option<DVectorViewMut<'b, f64>>; Q::LAST as usize]) -> Self{
        Self { data }
    }
    pub fn is_some(&self, q: Q) -> bool {
        self.data[q as usize].is_some()
    }
    pub fn is_none(&self, q: Q) -> bool {
        self.data[q as usize].is_none()
    }
    pub fn get_data(&mut self, q: Q) -> &DVectorViewMut<f64> {
        self.data[q as usize].as_mut().unwrap()
    }
    pub fn set_scalar(&mut self, q: Q, i: usize, value: f64) {
        self.data[q as usize].as_mut().unwrap()[i] = value;
    }
    pub fn add_scalar(&mut self, q: Q, i: usize, value: f64) {
        self.data[q as usize].as_mut().unwrap()[i] += value;
    }
    pub fn mul_scalar(&mut self, q: Q, i: usize, value: f64) {
        self.data[q as usize].as_mut().unwrap()[i] *= value;
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
    SquareTensor(usize),
    Tensor(usize, usize),
}

impl QDim {
    pub const fn size(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(n) => *n,
            QDim::SquareTensor(n) => n.pow(2),
            QDim::Tensor(n, m) => *n * *m,
        }
    }
    pub const fn dim(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(n) => *n,
            QDim::SquareTensor(n) => *n,
            QDim::Tensor(n, m) => panic!("Tensor dimensions are not implemented yet."),
        }
    }
}

impl Q {
    //TODO: When the const generics are stabilized, all functions should be made dependent on
    // an enumertion of constraints, e.g. 3D, PlaneStrain, PlaneStress etc.
    pub const fn q_dim(&self) -> QDim {
        match self {
            Q::MandelStress => QDim::Vector(6),
            Q::MandelStrain => QDim::Vector(6),
            Q::MandelStrainRate => QDim::Vector(6),
            Q::MandelTangent => QDim::SquareTensor(6),
            Q::VelocityGradient => QDim::SquareTensor(3),
            Q::NonlocalEquivalentStrain => QDim::Scalar,
            Q::Lambda => QDim::Scalar,
            Q::Density => QDim::Scalar,
            Q::Pressure => QDim::Scalar,
            Q::Damage => QDim::Scalar,
            Q::StrainRateNorm => QDim::Scalar,
            Q::EquivalentPlasticStrain => QDim::Scalar,
            Q::MisesStress => QDim::Scalar,
            Q::LAST => QDim::Scalar,
        }
    }
    pub const fn dim(&self) -> usize {
        self.q_dim().dim()
        
    }
    pub const fn size(&self) -> usize {
        self.q_dim().size()
    }

}

pub trait ConstitutiveModel {
    fn new(parameters: &HashMap<String, f64>) -> Self;
    //Mainly for the purpose of telling Python what is needed
    fn define_input(&self) -> HashMap<Q, QDim>;
    //Mainly for the purpose of telling Python what is needed
    fn define_output(&self) -> HashMap<Q, QDim>;
    fn define_optional_output(&self) -> HashMap<Q, QDim> {
        HashMap::new()
    }

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
