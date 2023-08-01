use nalgebra::{DVectorView, DVectorViewMut};

pub mod linear_elastic;

pub trait SmallStrainModel {
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
