import basix
import comfe as co
import dolfinx as df
import matplotlib.pyplot as plt
import numpy as np
import ufl
from mpi4py import MPI
from pint import Quantity


def dict_to_base_units_and_other(dic):
    new_dic = {}
    for key, value in dic.items():
        # try:
        if isinstance(value, list):
            new_dic[key] = Quantity(*value).to_base_units().magnitude
        else:
            new_dic[key] = value
        # except:
        #    new_dic[key] = value
    return new_dic


def dict_to_base_units(dic):
    new_dic = {}
    for key, value in dic.items():
        # try:
        if isinstance(value, list):
            new_dic[key] = Quantity(*value).to_base_units().magnitude
        # except:
        #    new_dic[key] = value
    return new_dic


def test_uniaxial_strain_3D(model, parameters) -> None:
    mesh = df.mesh.create_unit_cube(
        MPI.COMM_SELF,
        1,
        1,
        1,
        cell_type=df.mesh.CellType.hexahedron,
    )
    rule = co.helpers.QuadratureRule(
        quadrature_type=basix.QuadratureType.Default,
        cell_type=basix.CellType.hexahedron,
        degree=1,
    )
    parameters_nl = {
        "l": [0.0, "mm"],
        "zeta": [1e-5, "ms**2"],
        "gamma": [1e-3, "ms"],
        "decreasing_interaction": False,
        "eta": [5.0, ""],
        "R": [0.0, ""],
        "solver": "cdm",
        "initial_config": False,
    }
    parameters_nl = dict_to_base_units(parameters_nl)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    P1 = df.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    P1s = df.fem.FunctionSpace(mesh, ("Lagrange", 1))
    eps_nl = df.fem.Function(P1s)
    u = df.fem.Function(P1)
    t_end = 0.1
    t_pre = t_end / 100.0
    u_end = 0.0125
    v_end = u_end / t_end

    def v(t):
        if t > t_pre:
            return v_end
        else:
            return v_end * (t / t_pre)

    v_bc = df.fem.Constant(mesh, v_end / t_end)
    domains = [
        lambda x: np.isclose(x[0], 0.0),
        lambda x: np.isclose(x[0], 1.0),
        lambda x: np.isclose(x[1], 0.0),
        lambda x: np.isclose(x[1], 1.0),
        lambda x: np.isclose(x[2], 0.0),
        lambda x: np.isclose(x[2], 1.0),
    ]
    values = [0.0, v_bc, 0.0, 0.0, 0.0, 0.0]
    subspaces = [0, 0, 1, 1, 2, 2]
    boundary_facets = [df.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, domain) for domain in domains]
    bc_dofs = [
        df.fem.locate_dofs_topological(P1.sub(i), mesh.topology.dim - 1, facet)
        for facet, i in zip(boundary_facets, subspaces)
    ]

    bcs = [df.fem.dirichletbc(value, dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]

    parameters = dict_to_base_units_and_other(parameters)
    law = model(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    h = t_end / 100000.0
    density = parameters["RHO"] if "RHO" in parameters else parameters["density"]
    mass_form = ufl.inner(u_, v_) * density * ufl.dx

    M = df.fem.petsc.assemble_matrix(df.fem.form(mass_form))
    M.assemble()
    ones = df.fem.Function(P1)
    with ones.vector.localForm() as ones_local:
        ones_local.set(1.0)
    M_action = M * ones.vector
    M_function = df.fem.Function(P1)
    M_function.vector.array[:] = M_action.array

    M_action.array[:] = 1.0 / M_action.array
    M_action.ghostUpdate()

    solver = co.cdm.CDMNonlocalMechanics(
        P1,
        P1s,
        0,
        None,
        bcs,
        law,
        parameters,
        parameters_nl,
        "equivalent_plastic_strain",
        None,
        "equivalent_nonlocal_plastic_strain",
        "equivalent_nonlocal_plastic_strain_rate",
        rule,
        nonlocal_initial_config=False,
    )
    solver.mechanics_solver.model.input["density"].vector.array[:] += density
    solver.mechanics_solver.model.output["density"].vector.array[:] += density
    stress = []
    t = []
    u_max = []
    damage = []
    eps_pl = []
    while solver.t < t_end:
        v_bc.value = v(solver.t)
        try:
            solver.step(h)
        except:
            break
        u_ = max(abs(solver.fields["u"].vector.array))
        u_max.append(u_)
        stress.append(solver.q_fields["mandel_stress"].vector.array.copy())
        t.append(solver.t)
        damage.append(solver.q_fields["damage"].vector.array.copy())
        eps_pl.append(solver.q_fields["equivalent_plastic_strain"].vector.array.copy())
        if np.NAN in stress[-1]:
            raise ValueError("NAN in stress array, check parameters or model implementation.")
        print(len(u_max))
    # print(u_max)
    # print(stress)
    stress = np.array(stress).reshape(-1, 6)
    mandel_identity = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    pressure = (stress @ mandel_identity).flatten() / 3.0
    stress_dev = stress - mandel_identity * pressure.reshape(-1, 1)
    sig_1 = stress[:, 0]
    s_eq = np.sqrt(3.0 / 2.0 * (stress_dev**2).sum(axis=1))
    plt.plot(t, s_eq, label="Equivalent Stress")
    plt.xlabel("Time [s]")
    plt.ylabel("Equivalent Stress [MPa]")
    plt.show()
    plt.plot(t, pressure, label="Pressure")
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [MPa]")
    plt.show()
    plt.plot(t, damage, label="Damage")
    plt.xlabel("Time [s]")
    plt.ylabel("Damage")
    plt.show()
    plt.plot(t, sig_1, label="sigma_axial")
    plt.xlabel("Time [s]")
    plt.ylabel("Stress axial direction [MPa]")
    plt.show()
    plt.plot(t, eps_pl, label="Equivalent Plastic Strain")
    plt.xlabel("Time [s]")
    plt.ylabel("Equivalent Plastic Strain")
    plt.show()

    plt.plot(pressure, s_eq, label="Stress vs Pressure")
    plt.xlabel("Pressure [MPa]")
    plt.ylabel("Equivalent Stress [MPa]")
    plt.show()
    with df.io.XDMFFile(MPI.COMM_SELF, "output.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(solver.fields["u"], t=solver.t)
    #     import matplotlib
    #     import matplotlib.pyplot as plt

    #     matplotlib.use("Agg", force=True)
    #     p_debug = np.linspace(0.0, 8.0, 100)
    #     # plt.plot(p_debug, y_i(p_debug, parameters))
    #     # plt.plot(p_debug, y_f(p_debug, parameters))
    #     plt.plot(p_, s_eq_)
    #     plt.scatter(test_case["points"][:, 0], test_case["points"][:, 1])
    #     plt.xlabel("Pressure [GPa]")
    #     plt.ylabel("Equiv. Stress [GPa]")
    #     plt.title("JH2 test")
    #     plt.savefig(f"{plot}.png")
    #     plt.clf()

    # points = np.hstack((p_, s_eq_))
    # tree = KDTree(points)
    # distances = tree.query(test_case["points"])
    # assert np.mean(distances[0] / np.max(np.abs(test_case["points"][:, 1]))) < 0.05    assert np.mean(distances[0] / np.max(np.abs(test_case["points"][:, 1]))) < 0.05


if __name__ == "__main__":
    # parameters_jh2 = {
    #     "RHO": [2.440e-6, "kg / mm**3"],
    #     "rho": [2.440e-6, "kg / mm**3"],
    #     "SHEAR_MODULUS": [12000, "MPa"],
    #     "A": [0.6304, ""],
    #     "B": [0.2101, ""],
    #     "C": [0.0, ""],
    #     "M": [0.8437, ""],
    #     "N": [0.8437, ""],
    #     "EPS0": [1.0, "1 / s"],
    #     "T": [3.4, "MPa"],
    #     "SIGMAHEL": [1005, "MPa"],
    #     "PHEL": [0.811, "GPa"],
    #     "D1": [0.04, ""],  # from JH concrete model
    #     "D2": [1.0, ""],  # from JH concrete model
    #     "K1": [16.667, "GPa"],
    #     "K2": [73.19, "GPa"],
    #     "K3": [-236.2, "GPa"],
    #     "BETA": [0.0, ""],
    #     "EFMIN": [0.01, ""],  # from JH concrete model
    #     "DMAX": [1.0, ""],
    #     "E_F": [0.002, ""],
    #     "E_0": [0.0025, ""],
    # }
    # test_uniaxial_strain_3D(co.laws.PyGradientJH23D, parameters_jh2)
    # parameters_rub = {
    #     "density": [2.440e-6, "kg / mm**3"],
    #     "rho": [2.440e-6, "kg / mm**3"],
    #     "shear_modulus": [12000.0, "MPa"],
    #     "mu": [12000.0, "MPa"],
    #     "kappa": [16.667, "MPa"],
    #     "bulk_modulus": [16.667, "GPa"],
    #     "a_y": [5111012.97882567, ""],
    #     "b_y": [1277753.2447064174, ""],
    #     "d_y": [15311012.97882567, ""],
    #     "a_r": [5100000.0, ""],
    #     "b_r": [424932.58248730964, ""],
    #     "e_f": [0.002, ""],
    #     "h": [100.0, ""],
    #     "alpha_0": [0.00, ""],
    # }
    # test_uniaxial_strain_3D(co.laws.PyGradientRUB3D, parameters_rub)
    # parameters_engelen = {
    #     "e_f": [0.002, ""],
    #     "density": [2.440e-6, "kg / mm**3"],
    #     "rho": [2.440e-6, "kg / mm**3"],
    #     "mu": [12000.0, "MPa"],
    #     "kappa": [16.667, "GPa"],
    #     "sigma_y": [40.0, "MPa"],
    #     "alpha_0": [0.0, ""],
    #     "h": [2000.0, "MPa"],
    # }
    # test_uniaxial_strain_3D(co.laws.PyUniaxialStressEngelen3D, parameters_engelen)
    parameters_drucker_prager = {
        "a_y": [5111012.97882567, ""],
        "b_y": [1277753.2447064174, ""],
        "d_y": [15311012.97882567, ""],
        "e_f": [0.002, ""],
        "h": [100.0, ""],
        "alpha_0": [0.0025, ""],
        "density": [2.440e-6, "kg / mm**3"],
        "rho": [2.440e-6, "kg / mm**3"],
        "shear_modulus": [12000.0, "MPa"],
        "mu": [12000.0, "MPa"],
        "bulk_modulus": [16.667, "GPa"],
        "kappa": [16.667, "GPa"],
        "radial_factor": [0.0, ""],
    }
    test_uniaxial_strain_3D(co.laws.PyDruckerPrager3D, parameters_drucker_prager)
