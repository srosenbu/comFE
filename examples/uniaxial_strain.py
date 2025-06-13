import basix
import comfe as co
import dolfinx as df
import numpy as np
import ufl


def test_uniaxial_strain_3D(model, parameters) -> None:
    mesh = df.mesh.create_unit_cube(
        cell_type=df.mesh.CellType.hexahedron,
    )
    rule = co.helpers.QuadratureRule(
        quadrature_type=basix.QuadratureType.Default,
        cell_type=basix.CellType.hexahedron,
        degree=1,
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    P1 = df.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u = df.fem.Function(P1)
    t_end = 100.0
    v_bc = -50.0 / t_end
    domains = [
        lambda x: np.isclose(x[0], 0.0),
        lambda x: np.isclose(x[0], 1000.0),
        lambda x: np.isclose(x[1], 0.0),
        lambda x: np.isclose(x[1], 1000.0),
    ]
    values = [0.0, 0.0, 0.0, v_bc]
    subspaces = [0, 0, 1, 1]
    boundary_facets = [df.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, domain) for domain in domains]
    bc_dofs = [
        df.fem.locate_dofs_topological(P1.sub(i), mesh.topology.dim - 1, facet)
        for facet, i in zip(boundary_facets, subspaces)
    ]

    bcs = [df.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]

    parameters = parameters
    law = model(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    h = 1e-1
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

    solver = co.cdm.CDMPlaneStrain(
        P1,
        0,
        None,
        bcs,
        M_function,
        law,
        rule,
    )
    solver.model.input["density"].vector.array[:] += density
    solver.model.output["density"].vector.array[:] += density
    stress = []

    while solver.t < t_end:
        solver.step(h)
        u_ = max(abs(solver.fields["u"].vector.array))
        stress.append(solver.q_fields["mandel_stress"].vector.array)

    values = [0.0, 0.0, 0.0, -v_bc]
    subspaces = [0, 0, 1, 1]

    bcs = [df.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]
    solver.bcs = bcs

    while solver.t < 2.0 * t_end:  # and counter <= 2000:
        solver.step(h)
        u_ = max(abs(solver.fields["u"].vector.array))
        p_.append(solver.q_fields["pressure"].vector.array[0])
        s_eq_.append(solver.q_fields["mises_stress"].vector.array[0])

    p_ = np.array(p_).reshape((-1, 1))
    s_eq_ = np.array(s_eq_).reshape((-1, 1))

    # if plot is not None:
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
