use nalgebra::{ArrayStorage, SMatrix, SVector};

pub const fn sym_id<const N: usize>() -> SVector<f64, N> {
    let mut data = [0.0; N];
    data[0] = 1.0;
    if N > 1 {
        data[1] = 1.0;
    }
    if N > 2 {
        data[2] = 1.0;
    }
    let array_storage = ArrayStorage::<f64, N, 1>([data]);
    let output = SVector::<f64, N>::from_array_storage(array_storage);
    output
}

pub const fn sym_id_outer_sym_id<const N: usize>() -> SMatrix<f64, N, N> {
    let mut data = [[0.0; N]; N];
    data[0][0] = 1.0;
    if N > 1 {
        data[0][1] = 1.0;
        data[1][0] = 1.0;
        data[1][1] = 1.0;
    }
    if N > 2 {
        data[0][2] = 1.0;
        data[2][0] = 1.0;
        data[1][2] = 1.0;
        data[2][1] = 1.0;
        data[2][2] = 1.0;
    }
    let storage = ArrayStorage::<f64, N, N>(data);
    SMatrix::<f64, N, N>::from_array_storage(storage)
}

pub const fn id<const N: usize>() -> SMatrix<f64, N, N> {
    let mut data = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        data[i][i] = 1.0;
        i += 1;
    }
    let storage = ArrayStorage::<f64, N, N>(data);
    SMatrix::<f64, N, N>::from_array_storage(storage)
}
pub const fn const_add_matrices<const N: usize, const M: usize>(
    a: SMatrix<f64, N, M>,
    b: SMatrix<f64, N, M>,
) -> SMatrix<f64, N, M> {
    let mut c_data = [[0.0; N]; M];
    let a_data = a.data.0;
    let b_data = b.data.0;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < M {
            c_data[i][j] = a_data[i][j] + b_data[i][j];
            j += 1;
        }
        i += 1;
    }
    let storage = ArrayStorage::<f64, N, M>(c_data);
    SMatrix::<f64, N, M>::from_array_storage(storage)
}

pub const fn const_scalar_mult<const N: usize, const M: usize>(
    a: SMatrix<f64, N, M>,
    scalar: f64,
) -> SMatrix<f64, N, M> {
    let mut c_data = [[0.0; N]; M];
    let a_data = a.data.0;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < M {
            c_data[i][j] = a_data[i][j] * scalar;
            j += 1;
        }
        i += 1;
    }
    let storage = ArrayStorage::<f64, N, M>(c_data);
    SMatrix::<f64, N, M>::from_array_storage(storage)
}

pub const fn projection_dev<const N: usize>() -> SMatrix<f64, N, N> {
    let sym_id_outer_sym_id_scaled = const_scalar_mult(sym_id_outer_sym_id::<N>(), -1.0 / 3.0);
    const_add_matrices(sym_id_outer_sym_id_scaled, id::<N>())
}

pub const SYM_ID_6: SVector<f64, 6> = sym_id::<6>();
pub const SYM_ID_4: SVector<f64, 4> = sym_id::<4>();
pub const SYM_ID_3: SVector<f64, 3> = sym_id::<3>();
pub const SYM_ID_1: SVector<f64, 1> = sym_id::<1>();

pub const SYM_ID_6_OUTER_SYM_ID_6: SMatrix<f64, 6, 6> = sym_id_outer_sym_id::<6>();
pub const SYM_ID_4_OUTER_SYM_ID_4: SMatrix<f64, 4, 4> = sym_id_outer_sym_id::<4>();
pub const SYM_ID_3_OUTER_SYM_ID_3: SMatrix<f64, 3, 3> = sym_id_outer_sym_id::<3>();
pub const SYM_ID_1_OUTER_SYM_ID_1: SMatrix<f64, 1, 1> = sym_id_outer_sym_id::<1>();

pub const ID_6: SMatrix<f64, 6, 6> = id::<6>();
pub const ID_4: SMatrix<f64, 4, 4> = id::<4>();
pub const ID_3: SMatrix<f64, 3, 3> = id::<3>();
pub const ID_1: SMatrix<f64, 1, 1> = id::<1>();

pub const PROJECTION_DEV_6: SMatrix<f64, 6, 6> = projection_dev::<6>();
pub const PROJECTION_DEV_4: SMatrix<f64, 4, 4> = projection_dev::<4>();
pub const PROJECTION_DEV_3: SMatrix<f64, 3, 3> = projection_dev::<3>();
pub const PROJECTION_DEV_1: SMatrix<f64, 1, 1> = projection_dev::<1>();
