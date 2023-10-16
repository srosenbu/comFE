import logging

import basix
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from scipy.linalg import eigvals

__all__ = [
    "QuadratureRule",
    "set_mesh_coordinates",
    "QuadratureEvaluator",
    "diagonal_mass",
    "critical_timestep",
    "critical_timestep_nonlocal",
]


class LogMixin(object):
    @property
    def logger(self):
        name = self.__class__.__module__
        return logging.getLogger(name)


# class Spaces:
#     _self = None
#     _spaces = None
#     _q_spaces

#     def __new__(cls):
#         if cls._self is None:
#             cls._self = super().__new__(cls)
#             cls._spaces = {}
#         return cls._self

#     def __init__(self):
#         pass

#     def __getitem__(self, key: tuple[str, int | tuple[int,int]]) -> df.fem.FunctionSpace:
#         return self._spaces[key]

#     #def __setitem__(self, key: tuple[str, int | tuple[int,int]], space: df.fem.FunctionSpace)->None:
#     #    if key in self._spaces:
#     #        raise Exception(f"A {key[0]} space of dimension {key[1]} already exists. Please remove the extra space from your code.")
#     #    else:
#     #        self._spaces[key] = space

#     def create_quadrature_space(self, dim: int | tuple[int, int], rule:QuadratureRule) ->


def set_mesh_coordinates(mesh, x, mode="set"):
    dim = mesh.geometry.dim
    array = x if type(x) == np.ndarray else x.x.array
    if mode == "set":
        mesh.geometry.x[:, :dim] = array.reshape(-1, dim)
    elif mode == "add":
        mesh.geometry.x[:, :dim] += array.reshape(-1, dim)


class QuadratureRule:
    def __init__(
        self,
        quadrature_type=basix.QuadratureType.Default,
        cell_type=basix.CellType.triangle,
        degree=1,
    ):
        self.quadrature_type = quadrature_type
        self.cell_type = cell_type
        self.degree = degree
        self.points, self.weights = basix.make_quadrature(self.quadrature_type, self.cell_type, self.degree)
        self.dx = ufl.dx(
            metadata={
                "quadrature_rule": self.quadrature_type.name,
                "quadrature_degree": self.degree,
            }
        )

    def create_quadrature_space(self, mesh):
        Qe = ufl.FiniteElement(
            "Quadrature",
            basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
        )

        return df.fem.FunctionSpace(mesh, Qe)

    def create_quadrature_vector_space(self, mesh, dim):
        Qe = ufl.VectorElement(
            "Quadrature",
            basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
            dim=dim,
        )

        return df.fem.FunctionSpace(mesh, Qe)

    def create_quadrature_tensor_space(self, mesh, shape):
        Qe = ufl.TensorElement(
            "Quadrature",
            basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
            shape=shape,
        )

        return df.fem.FunctionSpace(mesh, Qe)

    def number_of_points(self, mesh):
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local
        return self.num_cells * self.weights.size

    def create_quadrature_space_like(self, function_space: df.fem.FunctionSpace):
        element = function_space.ufl_element()
        if len(element.value_shape()) == 0:
            return self.create_quadrature_space(function_space.mesh)
        elif len(element.value_shape()) == 1:
            return self.create_quadrature_vector_space(function_space.mesh, element.value_shape()[0])
        elif len(element.value_shape()) == 2:
            return self.create_quadrature_tensor_space(function_space.mesh, element.value_shape())


def basix_cell_type_to_ufl(cell_type: basix.CellType) -> ufl.Cell:
    conversion = {
        basix.CellType.interval: ufl.interval,
        basix.CellType.triangle: ufl.triangle,
        basix.CellType.tetrahedron: ufl.tetrahedron,
        basix.CellType.quadrilateral: ufl.quadrilateral,
        basix.CellType.hexahedron: ufl.hexahedron,
    }
    return conversion[cell_type]


def ufl_cell_to_basix(cell: ufl.Cell) -> basix.CellType:
    conversion = {
        ufl.interval: basix.CellType.interval,
        ufl.triangle: basix.CellType.triangle,
        ufl.tetrahedron: basix.CellType.tetrahedron,
        ufl.quadrilateral: basix.CellType.quadrilateral,
        ufl.hexahedron: basix.CellType.hexahedron,
    }
    return conversion[cell]


class QuadratureEvaluator:
    def __init__(self, ufl_expression, mesh, quadrature_rule):
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local  # + map_c.num_ghosts
        try:
            assert map_c.num_ghosts == 0
        except AssertionError as e:
            print(f"Warning: In QuadratureEvaluator: There are {map_c.num_ghosts} Quadrature ghost points.")

        self.cells = np.arange(0, self.num_cells, dtype=np.int32)

        self.expr = df.fem.Expression(ufl_expression, quadrature_rule.points)

    def __call__(self, q=None):
        if q is None:
            return self.expr.eval(self.cells)
        elif type(q) == np.ndarray:
            self.expr.eval(self.cells, values=q.reshape(self.num_cells, -1))
        else:
            self.expr.eval(self.cells, values=q.vector.array.reshape(self.num_cells, -1))
            q.x.scatter_forward()


def project(v, V, dx, u=None):
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)
    a_proj = ufl.inner(dv, v_) * dx
    b_proj = ufl.inner(v, v_) * dx
    if u is None:
        solver = df.fem.petsc.LinearProblem(a_proj, b_proj)
        uh = solver.solve()
        return uh
    else:
        solver = df.fem.petsc.LinearProblem(a_proj, b_proj, u=u)
        solver.solve()


def diagonal_mass(function_space, rho, invert=True) -> df.fem.Function:
    cell_type = ufl_cell_to_basix(function_space.mesh.ufl_cell())
    if cell_type in [
        basix.CellType.interval,
        basix.CellType.quadrilateral,
        basix.CellType.hexahedron,
    ]:
        # do gll integration
        # todo:adapt for higher order elements
        p_degree_to_q_degree = {1: 1, 2: 2}
        V_degree = function_space.ufl_element().degree()

        degree = p_degree_to_q_degree[V_degree]

        rule = QuadratureRule(quadrature_type=basix.QuadratureType.gll, cell_type=cell_type, degree=degree)

        u_ = ufl.TestFunction(function_space)
        v_ = ufl.TrialFunction(function_space)
        mass_form = ufl.inner(u_, v_) * rho * rule.dx
        M_action = df.fem.Function(function_space)
        M = df.fem.petsc.assemble_matrix(df.fem.form(mass_form))
        M.assemble()
        M_action.vector.array[:] = M.getDiagonal().array[:]
    else:
        rule = QuadratureRule(quadrature_type=basix.QuadratureType.Default, cell_type=cell_type, degree=1)
        u_ = ufl.TestFunction(function_space)
        v_ = ufl.TrialFunction(function_space)
        mass_form = ufl.inner(u_, v_) * rho * rule.dx
        v_temp = df.fem.Function(function_space)
        ones = v_temp.vector.copy()
        M_action = df.fem.Function(function_space)
        M = df.fem.petsc.assemble_matrix(df.fem.form(mass_form))
        M.assemble()
        with ones.localForm() as ones_local:
            ones_local.set(1.0)
        M_action.vector.array = (M * ones).array
    if invert:
        M_action.vector.array[:] = 1.0 / M_action.vector.array
    return M_action


# class CriticalTimestep(mesh, rho, K, G, order=1):
#    pass


def critical_timestep_1d(l_e, E, rho, order=1):
    h_mesh = df.mesh.create_interval(MPI.COMM_SELF, np.array([0.0, l_e]), [1], cell_type=df.mesh.CellType.interval)
    h_P1 = df.fem.FunctionSpace(h_mesh, ("CG", order))
    h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
    K_form = df.fem.form(E * ufl.inner(ufl.grad(h_u), ufl.grad(h_v)) * ufl.dx)
    M_form = df.fem.form(rho * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        df.fem.petsc.assemble_matrix(K_form),
        df.fem.petsc.assemble_matrix(M_form),
    )
    h_K.assemble()
    h_M.assemble()
    h_M = np.array(h_M[:, :])
    h_K = np.array(h_K[:, :])
    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)

    h = 2.0 / max_eig**0.5
    return h


def critical_timestep(l_x, l_y, G, K, rho, cell_type=df.mesh.CellType.quadrilateral, order=1):
    # todo: implement other cell_types
    # cell_type=mesh.topology.cell_type
    if cell_type == df.mesh.CellType.triangle:
        h_mesh = df.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
            diagonal=df.cpp.mesh.DiagonalType.crossed,
        )
    elif cell_type == df.mesh.CellType.quadrilateral:
        h_mesh = df.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
        )
    else:
        raise TypeError('Cell type "' + str(cell_type) + '" is not yet supported')

    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        e = eps(v)
        return (K - (2.0 / 3.0) * G) * ufl.tr(e) * ufl.Identity(2) + 2.0 * G * e

    h_P1 = df.fem.VectorFunctionSpace(h_mesh, ("CG", order))
    h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
    K_form = df.fem.form(ufl.inner(eps(h_u), sigma(h_v)) * ufl.dx)
    M_form = df.fem.form(rho * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        df.fem.petsc.assemble_matrix(K_form),
        df.fem.petsc.assemble_matrix(M_form),
    )
    h_K.assemble()
    h_M.assemble()
    h_M = np.array(h_M[:, :])
    h_K = np.array(h_K[:, :])
    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)

    h = 2.0 / max_eig**0.5
    return h


def critical_timestep_nonlocal(l_x, l_y, l, zeta, cell_type=df.mesh.CellType.quadrilateral, rule=None, order=1):
    # todo: implement other cell_types
    if cell_type == df.mesh.CellType.triangle:
        h_mesh = df.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
            diagonal=df.cpp.mesh.DiagonalType.crossed,
        )
    elif cell_type == df.mesh.CellType.quadrilateral:
        h_mesh = df.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
        )
    else:
        raise TypeError('Cell type "' + str(cell_type) + '" is not yet supported')
    dx = rule.dx if rule is not None else ufl.dx
    h_P1 = df.fem.FunctionSpace(h_mesh, ("CG", order))
    h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
    K_form = df.fem.form((l**2 * ufl.inner(ufl.grad(h_u), ufl.grad(h_v)) + h_u * h_v) * ufl.dx)
    M_form = df.fem.form(zeta * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        df.fem.petsc.assemble_matrix(K_form),
        df.fem.petsc.assemble_matrix(M_form),
    )
    h_K.assemble()
    h_M.assemble()
    h_M = np.array(h_M[:, :])
    h_K = np.array(h_K[:, :])

    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)

    h = 2.0 / max_eig**0.5
    return h
