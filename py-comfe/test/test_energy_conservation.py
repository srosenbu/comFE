from typing import Callable

import basix
import comfe as co
import dolfinx as dfx
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from scipy.spatial import KDTree

jh2_parameters = {
    "RHO": 3.7e-6,
    "SHEAR_MODULUS": 90.16,
    "A": 0.93,
    "B": 0.31,
    "C": 0.0,
    "M": 0.6,
    "N": 0.6,
    "EPS0": 1.0,
    "T": 0.2,
    "SIGMAHEL": 2.0,
    "PHEL": 1.46,
    "D1": 0.00815,
    "D2": 1.0,
    "K1": 130.95,
    "K2": 0.0,
    "K3": 0.0,
    "BETA": 0.0,
    "EFMIN": 0.0,
}
elastic_parameters = {
    "RHO": 3.7e-6,
    "SHEAR_MODULUS": 90.16,
    "A": 1e200,
    "B": 1e200,
    "C": 0.0,
    "M": 0.0,
    "N": 0.0,
    "EPS0": 1.0,
    "T": 1e200,
    "SIGMAHEL": 2.0,
    "PHEL": 1.46,
    "D1": 0.00815,
    "D2": 1.0,
    "K1": 130.95,
    "K2": 0.0,
    "K3": 0.0,
    "BETA": 0.0,
    "EFMIN": 0.0,
}


@pytest.mark.parametrize(
    "test_case",
    [jh2_parameters, elastic_parameters],
)
def test_single_element_2d_dirichlet(test_case: dict, plot: str | None = None) -> None:
    # TODO
    pass


class ExternalForces:
    def __init__(
        self,
        function_space: dfx.fem.FunctionSpace,
        surface: Callable,
        pressure: Callable,
    ):
        mesh = function_space.mesh
        marker = 42
        value = dfx.fem.Constant(mesh, 0.0)
        facet_indices = dfx.mesh.locate_entities(mesh, mesh.topology.dim - 1, surface)
        facet_markers = np.full(len(facet_indices), marker).astype(np.int32)
        facet_tag = dfx.mesh.meshtags(mesh, mesh.topology.dim - 1, facet_indices, facet_markers)

        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
        n = ufl.FacetNormal(mesh)
        v_ = ufl.TestFunction(function_space)
        self.form = dfx.fem.form(value * ufl.dot(-n, v_) * ds(42))
        self.value = value
        self.f_ext = dfx.fem.Function(function_space, name="f_ext")
        self.pressure = pressure

    def __call__(self, t: float) -> dfx.fem.Function:
        self.value.value = self.pressure(t)
        with self.f_ext.vector.localForm() as f_local:
            f_local.set(0.0)
        dfx.fem.petsc.assemble_vector(self.f_ext.vector, self.form)
        return self.f_ext


@pytest.mark.parametrize(
    "test_case",
    [jh2_parameters, elastic_parameters],
)
def test_single_element_2d_external_force(test_case: dict, plot: str | None = None) -> None:
    mesh = dfx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0, 0], [1000.0, 1000.0]]),
        [1, 1],
        cell_type=dfx.mesh.CellType.quadrilateral,
    )
    rule = co.helpers.QuadratureRule(
        quadrature_type=basix.QuadratureType.Default,
        cell_type=basix.CellType.quadrilateral,
        degree=1,
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    P1 = dfx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u = dfx.fem.Function(P1)
    t_end = 100.0
    # v_bc = -50.0 / t_end
    p_max = 8
    domains = [
        lambda x: np.isclose(x[0], 0.0),
        lambda x: np.isclose(x[0], 1000.0),
        lambda x: np.isclose(x[1], 0.0),
        lambda x: np.isclose(x[1], 1000.0),
    ]
    values = [0.0, 0.0, 0.0]
    subspaces = [0, 0, 1]
    boundary_facets = [dfx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, domain) for domain in domains]
    bc_dofs = [
        dfx.fem.locate_dofs_topological(P1.sub(i), mesh.topology.dim - 1, facet)
        for facet, i in zip(boundary_facets[:-1], subspaces)
    ]

    forces = ExternalForces(P1, domains[-1], lambda t: p_max * t / t_end)

    bcs = [dfx.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]
    print(bcs)
    parameters = test_case
    law = co.laws.PyJH23D(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    h = 1e-3
    M_action = co.helpers.diagonal_mass(P1, parameters["RHO"], True)

    solver = co.cdm.CDMPlaneStrain(
        P1,
        0,
        forces,
        bcs,
        M_action,
        law,
        rule,
        additional_output=[
            "mises_stress",
            "pressure",
            "equivalent_plastic_strain",
            "internal_energy",
            "internal_elastic_energy",
            "internal_plastic_energy",
        ],
        calculate_total_energy=True,
    )
    solver.model.input["density"].vector.array[:] += parameters["RHO"]
    solver.model.output["density"].vector.array[:] += parameters["RHO"]
    t_ = []
    s_eq_ = []
    p_ = []
    E_kin = []
    E_total = []
    E_int = []
    E_int_elastic = []
    E_int_plastic = []

    energy_value = solver.q_fields["internal_energy"].copy()
    energy_form = dfx.fem.form(energy_value * solver.q_fields["density"] * rule.dx)

    kinetic_energy_form = dfx.fem.form(
        0.5 * ufl.inner(solver.fields["v"], solver.fields["v"]) * solver.q_fields["density"] * rule.dx
    )
    counter = 0
    v_old = 0
    while solver.t < t_end:
        solver.step(h)
        if counter % 1 == 0:
            t_.append(solver.t)
            u_ = max(abs(solver.fields["u"].vector.array))
            p_.append(solver.q_fields["pressure"].vector.array[0])
            s_eq_.append(solver.q_fields["mises_stress"].vector.array[0])
            energy_value.vector.array[:] = solver.q_fields["internal_energy"].vector.array
            E_int.append(dfx.fem.assemble_scalar(energy_form))
            energy_value.vector.array[:] = solver.q_fields["internal_elastic_energy"].vector.array
            E_int_elastic.append(dfx.fem.assemble_scalar(energy_form))
            energy_value.vector.array[:] = solver.q_fields["internal_plastic_energy"].vector.array
            E_int_plastic.append(dfx.fem.assemble_scalar(energy_form))

            E_total.append(solver.total_energy)
            v = 0.5 * (solver.fields["v"].vector.array + v_old)
            E_kin.append(0.5 * np.inner(solver.M.vector.array, v**2))
        v_old = solver.fields["v"].vector.array.copy()
        counter += 1
    # values = [0.0, 0.0, 0.0, -v_bc]
    # subspaces = [0, 0, 1, 1]

    # bcs = [dfx.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]
    # solver.bcs = bcs

    # while solver.t < 2.0 * t_end:  # and counter <= 2000:
    #     solver.step(h)
    #     u_ = max(abs(solver.fields["u"].vector.array))
    #     p_.append(solver.q_fields["pressure"].vector.array[0])
    #     s_eq_.append(solver.q_fields["mises_stress"].vector.array[0])

    p_ = np.array(p_)
    s_eq_ = np.array(s_eq_)
    E_total = np.array(E_total)
    E_int = np.array(E_int)
    E_int_elastic = np.array(E_int_elastic)
    E_int_plastic = np.array(E_int_plastic)

    np.testing.assert_allclose(E_int, E_int_elastic + E_int_plastic)
    # TODO: kin energy
    # if plot is not None:
    #     import matplotlib
    #     import matplotlib.pyplot as plt

    #     matplotlib.use("Agg", force=True)
    #     # p_debug = np.linspace(0.0, 8.0, 100)
    #     # plt.plot(p_debug, y_i(p_debug, parameters))
    #     # plt.plot(p_debug, y_f(p_debug, parameters))
    #     plt.plot(t_, E_total - E_int, label="E_kin")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Energy [J]")
    #     plt.title(f"JH2 test")
    #     plt.savefig(f"{plot}.png")
    #     plt.clf()


# if __name__ == "__main__":
#     test_single_element_2d_external_force(elastic_parameters, "test_elastic")
#     test_single_element_2d_external_force(jh2_parameters, "test_jh2")
