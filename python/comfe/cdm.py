# import warnings

import numpy as np
import ufl
from petsc4py import PETSc
import basix

import dolfinx as df
import ufl
from .helpers import (
    QuadratureRule,
    QuadratureEvaluator,
    set_mesh_coordinates,
)
from pydantic import BaseModel
from .laws import RustConstitutiveModel, ConstitutiveModel
from .comfe import jaumann_rotation
from typing import Callable
from abc import abstractmethod, ABC


class ExplicitMechanicsSolver(BaseModel, ABC):
    @abstractmethod
    def step(self, h:float) -> None:
        pass



class NonlocalInterface(BaseModel, ABC):
    Q_local: str
    Q_nonlocal: str

    @abstractmethod
    def step(self, h: float, p_l: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_quadrature_values(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_nodal_values(self) -> np.ndarray:
        pass


class CDMX3D(ExplicitMechanicsSolver):
    function_space: df.fem.FunctionSpace
    t: float
    del_t: float
    # rule: QuadratureRule
    external_forces: Callable | None = None
    f_int_form: df.fem.FormMetaClass
    L_evaluator: QuadratureEvaluator
    bcs: list[df.fem.DirichletBCMetaClass]
    M: df.fem.Function
    #nonlocal_var: NonlocalInterface | None = None
    model: ConstitutiveModel
    fields: dict[str, df.fem.Function]
    q_fields: dict[str, df.fem.Function]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        function_space: df.fem.FunctionSpace,
        t0: float,
        f_ext: Callable | None,
        bcs: list[df.fem.DirichletBCMetaClass],
        M: df.fem.Function,
        rust_model: RustConstitutiveModel,
        quadrature_rule: QuadratureRule,
        #nonlocal_var: NonlocalInterface | None = None,
        additional_output: list[str] | None = None,
    )->None:
        # self.del_t = None
        v = df.fem.Function(function_space, name="Velocity")
        u = df.fem.Function(function_space, name="Displacements")
        f = df.fem.Function(function_space, name="Forces")
        
        model = ConstitutiveModel(
            rust_model,
            quadrature_rule,
            function_space.mesh,
            None,
            additional_output,
        )

        stress = model["mandel_stress"]

        test_function = ufl.TestFunction(function_space)

        f_int_ufl = (
            -ufl.inner(
                self._as_mandel(ufl.sym(ufl.grad(test_function))),
                stress,
            )
            * quadrature_rule.dx
        )

        f_int_form = df.fem.form(f_int_ufl)

        L_evaluator = QuadratureEvaluator(
            self._as_3d_tensor(ufl.nabla_grad(v)),
            function_space.mesh,
            quadrature_rule,
        )

        super().__init__(
            function_space=function_space,
            t=t0,
            del_t=0.0,
            external_forces=f_ext,
            f_int_form=f_int_form,
            bcs=bcs,
            L_evaluator=L_evaluator,
            model=model,
            M=M,
            #nonlocal_var=nonlocal_var,
            fields={"u": u, "v": v, "f": f},
            q_fields=model.output,
        )

    def _as_3d_tensor(self, T: ufl.core.expr.Expr):
        return T

    def _as_mandel(self, T: ufl.core.expr.Expr):
        """
        T:
            Symmetric 3x3 tensor
        Returns:
            Vector representation of T with factor sqrt(2) for off diagonal components
        """
        factor = 2**0.5
        return ufl.as_vector(
            [
                T[0, 0],
                T[1, 1],
                T[2, 2],
                factor * T[1, 2],
                factor * T[0, 2],
                factor * T[0, 1],
            ]
        )

    def stress_update(self, h):
        L = self.model.input["velocity_gradient"]
        sigma = self.model.input["mandel_stress"]
        self.L_evaluator(L)

        jaumann_rotation(h, L.vector.array, sigma.vector.array)

        # if self.nonlocal_var is not None:
        #    input_list[
        #        self.nonlocal_var.Q_nonlocal
        #    ] = self.nonlocal_var.get_quadrature_values()

        # TODO: Is GhostUpdate really used correcxtly
        self.model.evaluate(h)
        self.model.update()
        # TODO
        # self.stress.x.scatter_forward()

    def step(self, h):
        del_t_mid = (h + self.del_t) / 2.0 if self.del_t != 0.0 else h
        self.del_t = h

        with self.fields["f"].vector.localForm() as f_local:
            f_local.set(0.0)
        df.fem.petsc.assemble_vector(self.fields["f"].vector, self.f_int_form)
        # TODO
        self.fields["f"].vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        if self.external_forces is not None:
            self.fields["f"].vector.array[:] += self.external_forces(
                self.t
            ).vector.array
            # TODO
            self.fields["f"].vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

        # given: v_n-1/2, x_n/u_n, a_n, f_int_n
        # Advance velocities and nodal positions in time
        # if self.damping is None:
        #    c1 = 1.0
        #    c2 = del_t_mid
        # else:
        #    c1 = (2.0 - self.damping * del_t_mid) / (2.0 + self.damping * del_t_mid)
        #    c2 = 2.0 * del_t_mid / (2.0 + self.damping * del_t_mid)

        self.fields["v"].vector.array[:] += (
            del_t_mid * self.M.vector.array * self.fields["f"].vector.array
        )

        df.fem.set_bc(self.fields["v"].vector, self.bcs)
        # ghost entries are needed
        self.fields["v"].x.scatter_forward()
        # use v.x instead of v.vector, since mesh update requires ghost entries
        du_half = (0.5 * self.del_t) * self.fields["v"].x.array

        set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")
        # if self.nonlocal_var is not None:
        #    self.nonlocal_var.step(
        #        self.del_t, self.law.get_internal_var(self.nonlocal_var.Q_local)
        #    )
        self.stress_update(self.del_t)

        self.fields["u"].x.array[:] += 2.0 * du_half

        set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")

        self.t += self.del_t


class CDMPlaneStrainX(CDMX3D):
    def _as_3d_tensor(self, T):
        return ufl.as_matrix(
            [[T[0, 0], T[0, 1], 0.0], [T[1, 0], T[1, 1], 0.0], [0.0, 0.0, 0.0]]
        )

    def _as_mandel(self, T):
        """
        T:
            Symmetric 2x2 tensor
        Returns:
            Vector representation of T with factor sqrt(2) for off diagonal components
        """
        T3d = self._as_3d_tensor(T)
        factor = 2**0.5
        return ufl.as_vector(
            [
                T3d[0, 0],
                T3d[1, 1],
                0.0,
                0.0,
                0.0,
                factor * T3d[0, 1],
            ]
        )

# class CDMNonlocalBalance(NonlocalInterface):
#     M: df.fem.Function
#     fields: dict[str, df.fem.Function]
#     q_fields: dict[str, df.fem.Function]

#     def __init__(
#         self,
#         Q_local: str,
#         Q_nonlocal: str,
#         t0: float,
#         function_space: df.fem.FunctionSpace,
#         M: df.fem.Function,
#         l: float,
#         zeta: float,
#         gamma: float,
#         quadrature_rule: QuadratureRule,
#         q_fields: dict[str, df.fem.Function] | None = None,
#     ):
#         # self.M = M
#         # self.l = l
#         # self.zeta = zeta
#         # self.gamma = gamma
#         # self.quadrature_rule = quadrature_rule

    
#         QS = quadrature_rule.create_quadrature_space(function_space.mesh)

#         p_l = df.fem.Function(self.QS)
#         p_nl_q = p_l

#         self.p_nl = df.fem.Function(function_space)
#         self.dp_nl = df.fem.Function(function_space)

#         fields = {"equivalent_nonlocal_strain": df.fem.Function(function_space)}
#         q_fields =  

#         test_function = ufl.TestFunction(function_space)
#         f_int_ufl = (
#             self.l**2 * ufl.inner(ufl.grad(self.p_nl), ufl.grad(test_function))
#             + self.p_nl * test_function
#         ) * self.quadrature_rule.dx
#         f_ext_ufl = self.p_l * test_function * self.quadrature_rule.dx

#         self.f_ufl = -f_int_ufl + f_ext_ufl

#         self.f_form = df.fem.form(self.f_ufl)
#         self.f = self.p_nl.vector.copy()
#         self.delta_t = df.fem.Constant(self.mesh, 0.0)
#         # self.p_evaluator = QuadratureEvaluator(
#         # self.delta_t * self.dp_nl, self.mesh, self.quadrature_rule
#         # )
#         self.p_evaluator = QuadratureEvaluator(
#             self.p_nl, self.mesh, self.quadrature_rule
#         )

#     # @profile
#     def _substep(self, h):
#         # we assume that in the last step, the substeps were the same
#         # (otherwise, we would need to adapt \Delta t_{n+1/2})
#         with self.f.localForm() as f_local:
#             f_local.set(0.0)

#         self.f.ghostUpdate()
#         # self.delta_t.value = h

#         df.fem.petsc.assemble_vector(self.f, self.f_form)
#         self.f.ghostUpdate()

#         c = self.gamma / self.zeta
#         c1 = (2.0 - c * h) / (2.0 + c * h)
#         c2 = 2.0 * h / (2.0 + c * h)

#         set_local(
#             self.dp_nl, c1 * get_local(self.dp_nl) + c2 * self.M.array * self.f.array
#         )
#         add_local(self.p_nl, h * get_local(self.dp_nl))

#     # @profile
#     def step(self, h, p, substeps=1):
#         h_sub = h / substeps
#         # we assume that the local plastic strain is constant on all substeps
#         set_local(self.p_l, p)
#         for i in range(substeps):
#             self._substep(h_sub)

#         self.p_evaluator(self.p_nl_q)
#         self.t += h

#     def get_quadrature_values(self):
#         return self.p_nl_q

#     def get_nodal_values(self):
#         return self.p_nl

# class ImplicitNonlocalVariable(NonlocalInterface):
#     """This class should work with all constraints"""

#     def __init__(
#         self,
#         Q_local,
#         Q_nonlocal,
#         t0,
#         function_space,
#         l,
#         quadrature_rule,
#     ):
#         super().__init__(Q_local, Q_nonlocal)
#         self.t = t0
#         self.mesh = function_space.mesh
#         self.l = l
#         self.quadrature_rule = quadrature_rule

#         self.QS = self.quadrature_rule.create_quadrature_space(self.mesh)

#         self.p_l = df.fem.Function(self.QS)
#         self.p_nl_q = self.p_l.vector.array.copy()

#         self.p_nl = df.fem.Function(function_space)

#         test_function = ufl.TestFunction(function_space)
#         trial_function = ufl.TrialFunction(function_space)
#         b_form = ufl.inner(test_function, self.p_l) * self.quadrature_rule.dx
#         A_form = (
#             ufl.inner(test_function, trial_function)
#             + self.l**2.0
#             * ufl.inner(ufl.grad(test_function), ufl.grad(trial_function))
#         ) * self.quadrature_rule.dx

#         self.problem = df.fem.petsc.LinearProblem(A_form, b_form, u=self.p_nl)
#         # p_nl_h = problem.solve()

#         self.p_evaluator = QuadratureEvaluator(
#             self.p_nl, self.mesh, self.quadrature_rule
#         )

#     def step(self, h, p, substeps=1):
#         # we assume that the local plastic strain is constant on all substeps
#         set_local(self.p_l, p)
#         self.problem.solve()

#         self.p_evaluator(self.p_nl_q)
#         self.t += h

#     def get_quadrature_values(self):
#         return self.p_nl_q

#     def get_nodal_values(self):
#         return self.p_nl


