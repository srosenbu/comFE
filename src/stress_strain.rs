use nalgebra::{SMatrix, SVector};
pub fn tensor_to_mandel(tensor:SMatrix<f64, 3,3>) -> SVector<f64, 6> {
    //TODO: This is a hack. Replace once https://github.com/rust-lang/rust/issues/57241 is solved
    const SQRT:f64 = 1.4142135623730951;
    let mut mandel = SVector::<f64, 6>::zeros();
    mandel[0] = tensor[(0,0)];
    mandel[1] = tensor[(1,1)];
    mandel[2] = tensor[(2,2)];
    mandel[3] = SQRT * tensor[(1,2)];
    mandel[4] = SQRT * tensor[(0,2)];
    mandel[5] = SQRT * tensor[(0,1)];
    return mandel;
}

pub fn mandel_to_tensor(mandel:SVector<f64, 6>) -> SMatrix<f64, 3,3> {
    //TODO: This is a hack. Replace once https://github.com/rust-lang/rust/issues/57241 is solved
    const FACTOR:f64 = 0.7071067811865475;
    SMatrix::<f64, 3,3>::new(
        mandel[0], FACTOR * mandel[5], FACTOR * mandel[4],
        FACTOR * mandel[5], mandel[1], FACTOR * mandel[3],
        FACTOR * mandel[4], FACTOR * mandel[3], mandel[2],
    )
}
pub const T_dev : SMatrix<f64, 6, 6> =  SMatrix::<f64, 6, 6>::new(
    2.0/3.0, -1.0/3.0, -1.0/3.0, 0.0, 0.0, 0.0,
    -1.0/3.0, 2.0/3.0, -1.0/3.0, 0.0, 0.0, 0.0,
    -1.0/3.0, -1.0/3.0, 2.0/3.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0., 1.0, 0.0,
    0.0, 0.0, 0.0, 0., 0.0, 1.0,
);

pub const T_vol: SVector<f64, 6> = SVector::<f64, 6>::new(
    1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0, 0.0, 0.0
);
pub const T_id: SVector<f64, 6> = SVector::<f64, 6>::new(
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0
);