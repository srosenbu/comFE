import math

import basix
import comfe as co
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI


def unit_circle_right(t: float, t_end):
    return np.array([math.cos(0.5 * math.pi * t / t_end), math.sin(0.5 * math.pi * t / t_end), 0.0])


def unit_circle_left(t: float, t_end):
    return np.array(
        [math.cos(0.5 * math.pi + 0.5 * math.pi * t / t_end), math.sin(0.5 * math.pi + 0.5 * math.pi * t / t_end), 0.0]
    )


def unit_circle_middle(t: float, t_end):
    return np.array(
        [
            math.cos(0.25 * math.pi + 0.5 * math.pi * t / t_end),
            math.sin(0.25 * math.pi + 0.5 * math.pi * t / t_end),
            0.0,
        ]
    )


def test_rotation_stretch(file_path: str | None = None, stretch: float = 0.5):
    """
      ---------
     /|       /|
    / |      / |
    --|------  |
    | |_____|  |
    | /     | /
    |/______|/

    """
    E = 2e6
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    parameters = {"mu": mu, "lambda": lam, "density": 1.0}

    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)

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

    def edge_0_0_z(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    def edge_0_1_z(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 1.0)

    def edge_1_0_z(x):
        return np.isclose(x[0], 1.0) & np.isclose(x[1], 0.0)

    def edge_1_1_z(x):
        return np.isclose(x[0], 1.0) & np.isclose(x[1], 1.0)

    dofs_0_0_z = df.fem.locate_dofs_geometrical(P1, edge_0_0_z)
    dofs_0_1_z = df.fem.locate_dofs_geometrical(P1, edge_0_1_z)
    dofs_1_0_z = df.fem.locate_dofs_geometrical(P1, edge_1_0_z)
    dofs_1_1_z = df.fem.locate_dofs_geometrical(P1, edge_1_1_z)

    value_0_0_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_0_1_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_1_0_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_1_1_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))

    bcs = [
        df.fem.dirichletbc(value_0_0_z, dofs_0_0_z, P1),
        df.fem.dirichletbc(value_0_1_z, dofs_0_1_z, P1),
        df.fem.dirichletbc(value_1_0_z, dofs_1_0_z, P1),
        df.fem.dirichletbc(value_1_1_z, dofs_1_1_z, P1),
    ]

    law = co.laws.PyHypoelasticity3D(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    mass_form = ufl.inner(u_, v_) * parameters["density"] * ufl.dx

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

    solver = co.cdm.CDM3D(
        P1,
        0,
        None,
        bcs,
        M_function,
        law,
        rule,
    )
    stress = []
    t = []

    h = 1e-1
    n_steps = int(t_end / h)
    angle = (math.pi / 2.0) / n_steps
    h = t_end / n_steps
    x0_0_1 = np.array([0.0, 1.0, 0.0])
    x0_1_0 = np.array([1.0, 0.0, 0.0])
    x0_1_1 = np.array([1.0, 1.0, 0.0])
    x1_0_1 = np.array([0.0, 0.0, 0.0])
    x1_1_0 = np.array([0.0, 0.0, 0.0])
    x1_1_1 = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    with df.io.XDMFFile(MPI.COMM_WORLD, f"{file_path}output.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    while solver.t < t_end:
        radius = 1.0 + stretch * ((solver.t + h) / t_end)
        # beta = math.asin(stretch / radius)
        alpha = 0.5 * math.pi * ((solver.t + h) / t_end) + 0.5 * math.pi
        normal = np.array([math.cos(alpha), math.sin(alpha), 0.0])

        x1_0_1 = unit_circle_left(solver.t + h, t_end)
        value_0_1_z.value = (x1_0_1 - x0_0_1) / h
        x0_0_1 = x1_0_1
        x1_1_0 = unit_circle_right(solver.t + h, t_end) * radius
        value_1_0_z.value = (x1_1_0 - x0_1_0) / h
        x0_1_0 = x1_1_0
        # x1_1_1 = 2**0.5 * unit_circle_middle(solver.t+h, t_end)
        x1_1_1 = x1_1_0 + normal
        value_1_1_z.value = (x1_1_1 - x0_1_1) / h
        x0_1_1 = x1_1_1
        solver.step(h)
        t.append(solver.t)
        stress.append(solver.q_fields["mandel_stress"].x.array.copy())
        # with df.io.XDMFFile(MPI.COMM_WORLD, f"{file_path}output.xdmf", "a") as xdmf:
        #    xdmf.write_function(solver.fields["u"], solver.t)
    import matplotlib.pyplot as plt

    stress = np.array(stress).reshape(-1, 6)
    plt.plot(t, stress[:, 0])
    plt.plot(t, stress[:, 1])
    plt.plot(t, stress[:, 2])
    plt.show()


def test_rotation(file_path: str | None = None):
    """
      ---------
     /|       /|
    / |      / |
    --|------  |
    | |_____|  |
    | /     | /
    |/______|/

    """

    parameters = {"mu": 42.0, "lambda": 0.2, "density": 1.0}

    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)

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

    def edge_0_0_z(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    def edge_0_1_z(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 1.0)

    def edge_1_0_z(x):
        return np.isclose(x[0], 1.0) & np.isclose(x[1], 0.0)

    def edge_1_1_z(x):
        return np.isclose(x[0], 1.0) & np.isclose(x[1], 1.0)

    dofs_0_0_z = df.fem.locate_dofs_geometrical(P1, edge_0_0_z)
    dofs_0_1_z = df.fem.locate_dofs_geometrical(P1, edge_0_1_z)
    dofs_1_0_z = df.fem.locate_dofs_geometrical(P1, edge_1_0_z)
    dofs_1_1_z = df.fem.locate_dofs_geometrical(P1, edge_1_1_z)

    value_0_0_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_0_1_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_1_0_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_1_1_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))

    bcs = [
        df.fem.dirichletbc(value_0_0_z, dofs_0_0_z, P1),
        df.fem.dirichletbc(value_0_1_z, dofs_0_1_z, P1),
        df.fem.dirichletbc(value_1_0_z, dofs_1_0_z, P1),
        df.fem.dirichletbc(value_1_1_z, dofs_1_1_z, P1),
    ]

    law = co.laws.PyHypoelasticity3D(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    mass_form = ufl.inner(u_, v_) * parameters["density"] * ufl.dx

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

    solver = co.cdm.CDM3D(
        P1,
        0,
        None,
        bcs,
        M_function,
        law,
        rule,
    )
    stress = []
    t = []

    h = 1e-1
    n_steps = int(t_end / h)
    angle = (math.pi / 2.0) / n_steps
    h = t_end / n_steps
    x0_0_1 = np.array([0.0, 1.0, 0.0])
    x0_1_0 = np.array([1.0, 0.0, 0.0])
    x0_1_1 = np.array([1.0, 1.0, 0.0])
    x1_0_1 = np.array([0.0, 0.0, 0.0])
    x1_1_0 = np.array([0.0, 0.0, 0.0])
    x1_1_1 = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    stretch = 0.0
    with df.io.XDMFFile(MPI.COMM_WORLD, f"{file_path}output.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    solver.model.input["mandel_stress"].x.array[:] = np.array([1.0, 0, 0, 0, 0, 0])
    while solver.t < t_end:
        radius = 1.0 + stretch * ((solver.t + h) / t_end)
        # beta = math.asin(stretch / radius)
        alpha = 0.5 * math.pi * ((solver.t + h) / t_end) + 0.5 * math.pi
        normal = np.array([math.cos(alpha), math.sin(alpha), 0.0])

        x1_0_1 = unit_circle_left(solver.t + h, t_end)
        value_0_1_z.value = (x1_0_1 - x0_0_1) / h
        x0_0_1 = x1_0_1
        x1_1_0 = unit_circle_right(solver.t + h, t_end) * radius
        value_1_0_z.value = (x1_1_0 - x0_1_0) / h
        x0_1_0 = x1_1_0
        # x1_1_1 = 2**0.5 * unit_circle_middle(solver.t+h, t_end)
        x1_1_1 = x1_1_0 + normal
        value_1_1_z.value = (x1_1_1 - x0_1_1) / h
        x0_1_1 = x1_1_1
        solver.step(h)
        t.append(solver.t)
        stress.append(solver.q_fields["mandel_stress"].x.array.copy())
        # with df.io.XDMFFile(MPI.COMM_WORLD, f"{file_path}output.xdmf", "a") as xdmf:
        #    xdmf.write_function(solver.fields["u"], solver.t)
    import matplotlib.pyplot as plt

    stress = np.array(stress).reshape(-1, 6)
    plt.plot(t, stress[:, 0])
    plt.plot(t, stress[:, 1])
    plt.plot(t, stress[:, 2])
    plt.show()


def test_pure_stretch(file_path: str | None = None, stretch: float = 0.5):
    """
      ---------
     /|       /|
    / |      / |
    --|------  |
    | |_____|  |
    | /     | /
    |/______|/

    """
    E = 2e6
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    parameters = {"mu": mu, "lambda": lam, "density": 1.0}

    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)

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

    def edge_0_0_z(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    def edge_0_1_z(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 1.0)

    def edge_1_0_z(x):
        return np.isclose(x[0], 1.0) & np.isclose(x[1], 0.0)

    def edge_1_1_z(x):
        return np.isclose(x[0], 1.0) & np.isclose(x[1], 1.0)

    dofs_0_0_z = df.fem.locate_dofs_geometrical(P1, edge_0_0_z)
    dofs_0_1_z = df.fem.locate_dofs_geometrical(P1, edge_0_1_z)
    dofs_1_0_z = df.fem.locate_dofs_geometrical(P1, edge_1_0_z)
    dofs_1_1_z = df.fem.locate_dofs_geometrical(P1, edge_1_1_z)

    value_0_0_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_0_1_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_1_0_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    value_1_1_z = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))

    bcs = [
        df.fem.dirichletbc(value_0_0_z, dofs_0_0_z, P1),
        df.fem.dirichletbc(value_0_1_z, dofs_0_1_z, P1),
        df.fem.dirichletbc(value_1_0_z, dofs_1_0_z, P1),
        df.fem.dirichletbc(value_1_1_z, dofs_1_1_z, P1),
    ]

    law = co.laws.PyHypoelasticity3D(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    mass_form = ufl.inner(u_, v_) * parameters["density"] * ufl.dx

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

    solver = co.cdm.CDM3D(
        P1,
        0,
        None,
        bcs,
        M_function,
        law,
        rule,
    )
    stress = []
    t = []

    h = 1e-1
    # n_steps = int(t_end / h)
    # h = t_end / n_steps
    v_x = stretch / t_end
    value_1_0_z.value = np.array([v_x, 0.0, 0.0])
    value_1_1_z.value = np.array([v_x, 0.0, 0.0])
    with df.io.XDMFFile(MPI.COMM_WORLD, f"{file_path}output.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    while solver.t < t_end:

        solver.step(h)
        t.append(solver.t)
        stress.append(solver.q_fields["mandel_stress"].x.array.copy())
        # with df.io.XDMFFile(MPI.COMM_WORLD, f"{file_path}output.xdmf", "a") as xdmf:
        #    xdmf.write_function(solver.fields["u"], solver.t)
    import matplotlib.pyplot as plt

    stress = np.array(stress).reshape(-1, 6)
    plt.plot(t, stress[:, 0])
    plt.plot(t, stress[:, 1])
    plt.plot(t, stress[:, 2])
    plt.show()


if __name__ == "__main__":
    test_rotation("output.xdmf")
    test_rotation_stretch("output.xdmf", stretch=0.01)
    test_pure_stretch("output.xdmf", stretch=0.01)

    # import matplotlib.pyplot as plt

    # stretch = 0.5
    # t = np.linspace(0, 100, 10)
    # circle_left = np.array([unit_circle_left(ti, 100) for ti in t]).reshape(-1, 3)
    # circle_right = np.array([unit_circle_right(ti, 100) for ti in t]).reshape(-1, 3)
    # circle_middle = np.array(
    #     [2**0.5 * unit_circle_middle(ti, 100) * (1 + stretch * (ti / 100)) for ti in t]
    # ).reshape(-1, 3)
    # stretched_circle_right = np.array([unit_circle_right(ti, 100) * (1 + stretch * (ti / 100)) for ti in t]).reshape(
    #     -1, 3
    # )
    # plt.plot(circle_left[:, 0], circle_left[:, 1], label="left")
    # plt.plot(circle_right[:, 0], circle_right[:, 1], label="right")
    # plt.plot(stretched_circle_right[:, 0], stretched_circle_right[:, 1], label="stretched right")
    # plt.plot(circle_middle[:, 0], circle_middle[:, 1], label="middle")
    # plt.legend()
    # plt.show()
