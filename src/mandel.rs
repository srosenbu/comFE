use nalgebra::{SMatrix, SVector, SVectorView};

pub fn trace(vector: &SVector<f64, 6>) -> f64 {
    vector.x + vector.y + vector.z
}
