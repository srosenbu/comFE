import comfe
import numpy as np

E = 42.0
nu = 0.3
lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
D = np.array([
    [lambda_ + 2.0 * mu, lambda_, lambda_, 0., 0., 0.],
    [lambda_, lambda_ + 2.0 * mu, lambda_, 0., 0., 0.],
    [lambda_, lambda_, lambda_ + 2.0 * mu, 0., 0., 0.],
    [0., 0., 0., 2.* mu, 0., 0.],
    [0., 0., 0., 0., 2.* mu, 0.],
    [0., 0., 0., 0., 0., 2.* mu],
    ])
parameters = {"E": E, "nu": nu}
model1 = comfe.PyLinearElastic3D(parameters)
model2 = comfe.PyLinElas3D(parameters)



def test_linear_elasticity() -> None:
    n = 42
    strain = np.random.random(6*n)
    stress_rust = np.zeros(6*n)
    tangent_rust = np.zeros(36*n)
    stress_python = strain.reshape(-1, 6) @ D
    tangent_python = np.tile(D.ravel(), n)
    model1.evaluate(1.0, stress_rust, strain, tangent_rust)
    np.testing.assert_allclose(stress_rust, stress_python.ravel())
    np.testing.assert_allclose(tangent_rust, tangent_python)
    model2.evaluate(1.0, {"mandel_strain": strain}, {"mandel_stress": stress_rust, "mandel_tangent": tangent_rust})
    np.testing.assert_allclose(stress_rust, stress_python.ravel())
    np.testing.assert_allclose(tangent_rust, tangent_python)


