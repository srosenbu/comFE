from comfe import jaumann_rotation, jaumann_rotation_expensive
import numpy as np


def jaumann_rotate_3d_python(L, sigma, del_t):
    W = 0.5 * (L - L.T)
    sigma_out = np.array(
        [
            [sigma[0], sigma[5] / 2**0.5, sigma[4] / 2**0.5],
            [sigma[5] / 2**0.5, sigma[1], sigma[3] / 2**0.5],
            [sigma[4] / 2**0.5, sigma[3] / 2**0.5, sigma[2]],
        ]
    )
    sigma_out += del_t * (sigma_out @ W.T + W @ sigma_out)
    return np.array(
        [
            sigma_out[0, 0],
            sigma_out[1, 1],
            sigma_out[2, 2],
            2**0.5 * sigma_out[1, 2],
            2**0.5 * sigma_out[0, 2],
            2**0.5 * sigma_out[0, 1],
        ]
    )


def test_jaumann_rotation():
    n = 42
    L = np.arange(9 * n, dtype=np.float64)
    sigma = np.arange(6 * n, dtype=np.float64)
    del_t = 0.1
    sigma_python = np.array(
        [
            jaumann_rotate_3d_python(L_i.reshape(3, 3), sigma_i, del_t)
            for L_i, sigma_i in zip(L.reshape(-1, 9), sigma.reshape(-1, 6))
        ]
    ).flatten()
    jaumann_rotation_expensive(del_t, L, sigma)
    np.testing.assert_allclose(sigma, sigma_python)
    sigma = np.arange(6 * n, dtype=np.float64)
    jaumann_rotation(del_t, L, sigma)
    np.testing.assert_allclose(sigma, sigma_python)
