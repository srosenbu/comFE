import math
from abc import ABC, abstractmethod
from typing import Callable, Type

import dolfinx as df
import numpy as np
import ufl
from dolfinx.cpp.la import ScatterMode
from petsc4py import PETSc
from pydantic import BaseModel

from .comfe import jaumann_rotation  # , jaumann_rotation_expensive
from .helpers import LogMixin, QuadratureEvaluator, QuadratureRule, diagonal_mass, set_mesh_coordinates
from .laws import ConstitutiveModel, RustConstitutiveModel

__all__ = [
    "ExplicitMechanicsSolver",
    "CDM3D",
    "CDMPlaneStrain",
    "CDMUniaxialStrain",
    "CDMNonlocalMechanics",
    "CDMSolver",
    "NonlocalInterface",
    "CDMNonlocal",
    "ImplicitNonlocal",
]

# class BulkViscosity(BaseModel):
#     c_l: float = 0.06
#     c_0: float = 1.5
#     expr: df.fem.Expression
#     diameter: df.fem.Function

#     class Config:
#         arbitrary_types_allowed = True

#     @staticmethod
#     def from_rule(mesh: df.mesh.Mesh, rule:QuadratureRule)->None:
#         #dim = mesh.geometry.dim
#         cell_space = rule.create_quadrature_space(mesh)
#         diameter = df.fem.Function(cell_space, name="diameter")
#         expr = df.fem.Expression(ufl.CellDiameter(mesh), cell_space.element.interpolation_points())
#         super().__init__(expr=expr, diameter=diameter)

#     def evaluate(self):
#         self.diameter.interpolate(self.expr)
#         self.diameter.x.scatter_forward()


class ExplicitMechanicsSolver(BaseModel, ABC):
    @abstractmethod
    def step(self, h: float) -> None:
        pass


class CDMSolver(BaseModel, ABC, LogMixin):
    fields: dict[str, df.fem.Function]
    q_fields: dict[str, df.fem.Function | ufl.core.expr.Expr]

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def step(self, h: float) -> None:
        pass


class NonlocalInterface(CDMSolver):
    Q_local: str
    Q_local_damage: str | None
    Q_nonlocal: str
    Q_nonlocal_rate: str


class CDM3D(CDMSolver):
    function_space: df.fem.FunctionSpace
    t: float
    del_t: float
    # rule: QuadratureRule
    external_forces: Callable | None = None
    f_int_form: df.fem.FormMetaClass
    L_evaluator: QuadratureEvaluator
    bcs: list[df.fem.DirichletBCMetaClass]
    M: df.fem.Function
    model: ConstitutiveModel
    # fields: dict[str, df.fem.Function]
    # q_fields: dict[str, df.fem.Function]
    total_energy: float | None = None
    distortion_expr: df.fem.Expression | None = None
    viscosity_evaluator: QuadratureEvaluator | None = None
    # energy_form: df.fem.FormMetaClass | None = None

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
        additional_output: list[str] | None = None,
        calculate_total_energy: bool = False,
        calculate_bulk_viscosity: bool = False,
        monitor_distortion: bool = False,
    ) -> None:
        # self.del_t = None
        v = df.fem.Function(function_space, name="Velocity")
        u = df.fem.Function(function_space, name="Displacements")
        f = df.fem.Function(function_space, name="Forces")
        fields = {"u": u, "v": v, "f": f}
        if monitor_distortion:
            detJ_space = df.fem.FunctionSpace(function_space.mesh, ("DG", 0))
            detJ = df.fem.Function(detJ_space, name="detJ")
            distortion_expr = df.fem.Expression(
                ufl.JacobianDeterminant(detJ_space.mesh), detJ_space.element.interpolation_points()
            )
            fields["detJ"] = detJ
        else:
            distortion_expr = None

        model = ConstitutiveModel(
            rust_model,
            quadrature_rule,
            function_space.mesh,
            None,
            additional_output,
        )

        stress = model["mandel_stress"]

        test_function = ufl.TestFunction(function_space)
        try:
            f_int_ufl = (
                -ufl.inner(
                    self._as_mandel(ufl.sym(ufl.grad(test_function))),
                    stress,
                )
                * quadrature_rule.dx
            )
        except:
            f_int_ufl = (
                -ufl.inner(
                    self._as_mandel(ufl.grad(test_function)),
                    stress,
                )
                * quadrature_rule.dx
            )

        f_int_form = df.fem.form(f_int_ufl)
        try:
            L_evaluator = QuadratureEvaluator(
                self._as_3d_tensor(ufl.nabla_grad(v)),
                function_space.mesh,
                quadrature_rule,
            )
        except:
            L_evaluator = QuadratureEvaluator(
                self._as_3d_tensor(ufl.grad(v)),
                function_space.mesh,
                quadrature_rule,
            )
        if calculate_bulk_viscosity:
            assert "cell_diameter" in model.input
            viscosity_evaluator = QuadratureEvaluator(
                ufl.CellDiameter(function_space.mesh),
                function_space.mesh,
                quadrature_rule,
            )
        else:
            viscosity_evaluator = None

        if calculate_total_energy:
            self.logger.warning(
                "Total energy calculation is expensive and should only be used for debugging purposes."
            )
            total_energy = 0.0
        else:
            total_energy = None

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
            # nonlocal_var=nonlocal_var,
            fields=fields,
            q_fields=model.output,
            total_energy=total_energy,
            distortion_expr=distortion_expr,
            viscosity_evaluator=viscosity_evaluator,
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
        if self.viscosity_evaluator is not None:
            cell_diameter = self.model.input["cell_diameter"]
            self.viscosity_evaluator(cell_diameter)

        self.L_evaluator(L)
        # print(sigma.vector.array)

        jaumann_rotation(h, L.vector.array, sigma.vector.array)

        # if self.nonlocal_var is not None:
        #    input_list[
        #        self.nonlocal_var.Q_nonlocal
        #    ] = self.nonlocal_var.get_quadrature_values()

        # TODO: Is GhostUpdate really used correcxtly
        self.model.evaluate(h)
        self.model.update()

    def step(self, h, intermediate_step: Callable | None = None) -> None:
        del_t_mid = (h + self.del_t) / 2.0 if self.del_t != 0.0 else h
        self.del_t = h

        with self.fields["f"].vector.localForm() as f_local:
            f_local.set(0.0)
        df.fem.petsc.assemble_vector(self.fields["f"].vector, self.f_int_form)
        self.fields["f"].x.scatter_reverse(ScatterMode.add)

        if self.external_forces is not None:
            external_forces = self.external_forces(self.t)
            self.fields["f"].vector.array[:] += external_forces.vector.array
            self.fields["f"].x.scatter_forward()

        # given: v_n-1/2, x_n/u_n, a_n, f_int_n
        # Advance velocities and nodal positions in time
        # if self.damping is None:
        #    c1 = 1.0
        #    c2 = del_t_mid
        # else:
        #    c1 = (2.0 - self.damping * del_t_mid) / (2.0 + self.damping * del_t_mid)
        #    c2 = 2.0 * del_t_mid / (2.0 + self.damping * del_t_mid)

        self.fields["v"].vector.array[:] += del_t_mid * self.M.vector.array * self.fields["f"].vector.array
        self.fields["v"].x.scatter_forward()

        df.fem.set_bc(self.fields["v"].vector, self.bcs)
        # ghost entries are needed
        self.fields["v"].x.scatter_forward()
        # use v.x instead of v.vector, since mesh update requires ghost entries

        du_half = (0.5 * self.del_t) * self.fields["v"].x.array

        set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")

        # basically, evaluate the nonlocal variable here
        if intermediate_step is not None:
            intermediate_step(h)

        self.stress_update(self.del_t)

        self.fields["u"].x.array[:] += 2.0 * du_half
        self.fields["u"].x.scatter_forward()

        set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")

        if self.total_energy is not None:
            external_forces_0 = external_forces.vector.array.copy()

            # undo mesh update
            # set_mesh_coordinates(self.function_space.mesh, -du_half, mode="add")
            external_forces = self.external_forces(self.t + self.del_t)

            external_forces.vector.array[:] = 0.5 * (external_forces_0 + external_forces.vector.array)

            # energy_form = df.fem.form(ufl.inner(external_forces, self.fields["v"]) * ufl.ds)
            # energy_increment = df.fem.assemble_scalar(energy_form)
            energy_increment = np.inner(external_forces.vector.array, self.fields["v"].vector.array)
            self.total_energy += self.del_t * energy_increment
            # redo mesh update
            # set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")
        #
        self.t += self.del_t


class CDMPlaneStrain(CDM3D):
    def _as_3d_tensor(self, T):
        return ufl.as_matrix([[T[0, 0], T[0, 1], 0.0], [T[1, 0], T[1, 1], 0.0], [0.0, 0.0, 0.0]])

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


class CDMUniaxialStrain(CDM3D):
    def _as_3d_tensor(self, T):
        return ufl.as_matrix([[T[0], 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def _as_mandel(self, T):
        """
        T:
            Symmetric 2x2 tensor
        Returns:
            Vector representation of T with factor sqrt(2) for off diagonal components
        """
        print(T.ufl_shape)
        T3d = self._as_3d_tensor(T)
        return ufl.as_vector(
            [
                T3d[0, 0],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )


class CDMNonlocal(NonlocalInterface):
    M: df.fem.Function
    # fields: dict[str, df.fem.Function]
    # q_fields: dict[str, df.fem.Function | ufl.core.expr.Expr]
    strain_evaluator: QuadratureEvaluator
    parameters: dict[str, float]
    form: df.fem.FormMetaClass
    t: float

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        Q_local: str,
        Q_nonlocal_rate: str,
        t0: float,
        function_space: df.fem.FunctionSpace,
        M: df.fem.Function,
        parameters: dict[str, float],
        quadrature_rule: QuadratureRule,
        q_fields_local: dict[str, df.fem.Function],
        q_field_nonlocal: df.fem.Function,
        Q_local_damage: str | None = None,
        displacements: df.fem.Function | None = None,
    ):
        q_fields = {Q_local: q_fields_local[Q_local]}

        if Q_nonlocal_rate[-4:] != "rate":
            raise ValueError("Q_nonlocal_rate must be some rate for CDM, you provided " + Q_nonlocal_rate)

        Q_nonlocal = Q_nonlocal_rate[:-5]
        q_fields[Q_nonlocal] = q_field_nonlocal

        fields = {
            Q_nonlocal_rate: df.fem.Function(function_space, name=Q_nonlocal_rate),
            Q_nonlocal: df.fem.Function(function_space, name=Q_nonlocal),
            "nonlocal_force": df.fem.Function(function_space, name="nonlocal_force"),
        }

        test_function = ufl.TestFunction(function_space)

        if Q_local_damage is not None:
            q_fields[Q_local_damage] = q_fields_local[Q_local_damage]

            eta, R = parameters["eta"], parameters["R"]
            g = ((1.0 - R) * ufl.exp(-eta * q_fields[Q_local_damage]) + R - math.exp(-eta)) / (1.0 - math.exp(-eta))
            q_fields["interaction"] = g
        else:
            g = 1.0

        if displacements is not None:
            fields["u"] = displacements

        f_int_ufl = (
            g * parameters["l"] ** 2 * ufl.inner(ufl.grad(fields[Q_nonlocal]), ufl.grad(test_function))
            + fields[Q_nonlocal] * test_function
        ) * quadrature_rule.dx

        f_ext_ufl = q_fields[Q_local] * test_function * quadrature_rule.dx

        f_ufl = -f_int_ufl + f_ext_ufl

        f_form = df.fem.form(f_ufl)

        strain_evaluator = QuadratureEvaluator(fields[Q_nonlocal], function_space.mesh, quadrature_rule)
        super().__init__(
            Q_local=Q_local,
            Q_nonlocal=Q_nonlocal,
            Q_local_damage=Q_local_damage,
            Q_nonlocal_rate=Q_nonlocal_rate,
            t=t0,
            M=M,
            parameters=parameters,
            q_fields=q_fields,
            fields=fields,
            form=f_form,
            strain_evaluator=strain_evaluator,
        )

    def step(self, h: float) -> None:
        with self.fields["nonlocal_force"].vector.localForm() as f_local:
            f_local.set(0.0)

        if "u" in self.fields:
            set_mesh_coordinates(self.fields["u"].function_space.mesh, -self.fields["u"].x.array, mode="add")
            # self.fields["u"].function_space.mesh.geometry.x[:] -= self.fields["u"].x.array

        df.fem.petsc.assemble_vector(self.fields["nonlocal_force"].vector, self.form)
        self.fields["nonlocal_force"].x.scatter_reverse(ScatterMode.add)

        c = self.parameters["gamma"] / self.parameters["zeta"]
        c1 = (2.0 - c * h) / (2.0 + c * h)
        c2 = 2.0 * h / (2.0 + c * h)

        self.fields[self.Q_nonlocal_rate].vector.array[:] = (
            c1 * self.fields[self.Q_nonlocal_rate].vector.array
            + c2 * self.M.vector.array * self.fields["nonlocal_force"].vector.array
        )
        self.fields[self.Q_nonlocal_rate].x.scatter_forward()

        self.fields[self.Q_nonlocal].vector.array[:] += h * self.fields[self.Q_nonlocal_rate].vector.array
        self.fields[self.Q_nonlocal].x.scatter_forward()

        self.strain_evaluator(self.q_fields[self.Q_nonlocal])
        self.q_fields[self.Q_nonlocal].x.scatter_forward()

        if "u" in self.fields:
            set_mesh_coordinates(self.fields["u"].function_space.mesh, self.fields["u"].x.array, mode="add")
            # self.fields["u"].function_space.mesh.geometry.x[:] += self.fields["u"].x.array

        self.t += h


class ImplicitNonlocal(NonlocalInterface):
    # M: df.fem.Function
    # fields: dict[str, df.fem.Function]
    # q_fields: dict[str, df.fem.Function | ufl.core.expr.Expr]
    strain_evaluator: QuadratureEvaluator
    parameters: dict[str, float]
    # form: df.fem.FormMetaClass
    problem: df.fem.petsc.LinearProblem
    t: float

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        Q_local: str,
        Q_nonlocal_rate: str,
        t0: float,
        function_space: df.fem.FunctionSpace,
        M: df.fem.Function,
        parameters: dict[str, float],
        quadrature_rule: QuadratureRule,
        q_fields_local: dict[str, df.fem.Function],
        q_field_nonlocal: df.fem.Function,
        Q_local_damage: str | None = None,
        displacements: df.fem.Function | None = None,
    ):
        q_fields = {Q_local: q_fields_local[Q_local]}

        if Q_nonlocal_rate[-4:] != "rate":
            raise ValueError("Q_nonlocal_rate must be some rate for CDM, you provided " + Q_nonlocal_rate)

        # q_fields[Q_nonlocal_rate] = q_field_nonlocal_rate

        Q_nonlocal = Q_nonlocal_rate[:-5]
        q_fields[Q_nonlocal] = q_field_nonlocal

        fields = {
            # Q_nonlocal_rate: df.fem.Function(function_space, name=Q_nonlocal_rate),
            Q_nonlocal: df.fem.Function(function_space, name=Q_nonlocal),
            "nonlocal_force": df.fem.Function(function_space, name="nonlocal_force"),
        }

        if Q_local_damage is not None:
            q_fields[Q_local_damage] = q_fields_local[Q_local_damage]

            eta, R = parameters["eta"], parameters["R"]
            g = ((1.0 - R) * ufl.exp(-eta * q_fields[Q_local_damage]) + R - math.exp(-eta)) / (1.0 - math.exp(-eta))
            q_fields["interaction"] = g
        else:
            g = 1.0

        if displacements is not None:
            fields["u"] = displacements

        test_function = ufl.TestFunction(function_space)
        trial_function = ufl.TrialFunction(function_space)

        b_form = ufl.inner(test_function, q_fields[Q_local]) * quadrature_rule.dx

        A_form = (
            ufl.inner(test_function, trial_function)
            + g * parameters["l"] ** 2.0 * ufl.inner(ufl.grad(test_function), ufl.grad(trial_function))
        ) * quadrature_rule.dx

        problem = df.fem.petsc.LinearProblem(A_form, b_form, u=fields[Q_nonlocal])

        # f_int_ufl = (
        #    g * parameters["l"] ** 2 * ufl.inner(ufl.grad(fields[Q_nonlocal]), ufl.grad(test_function))
        #    + fields[Q_nonlocal] * test_function
        # ) * quadrature_rule.dx

        # f_ext_ufl = q_fields[Q_local] * test_function * quadrature_rule.dx

        # f_ufl = -f_int_ufl + f_ext_ufl

        # f_form = df.fem.form(f_ufl)

        strain_evaluator = QuadratureEvaluator(fields[Q_nonlocal], function_space.mesh, quadrature_rule)
        super().__init__(
            Q_local=Q_local,
            Q_nonlocal=Q_nonlocal,
            Q_local_damage=Q_local_damage,
            Q_nonlocal_rate=Q_nonlocal_rate,
            t=t0,
            M=M,
            parameters=parameters,
            q_fields=q_fields,
            fields=fields,
            # form=f_form,
            strain_evaluator=strain_evaluator,
            problem=problem,
        )

    def step(self, h: float) -> None:
        # old_nonlocal = self.fields[self.Q_nonlocal].vector.array.copy()
        if "u" in self.fields:
            set_mesh_coordinates(self.fields["u"].function_space.mesh, -self.fields["u"].x.array, mode="add")

        self.problem.solve()
        # self.q_fields[self.Q_nonlocal_rate].vector.array[:] = (
        #    self.fields[self.Q_nonlocal].vector.array - old_nonlocal
        # ) / h
        self.strain_evaluator(self.q_fields[self.Q_nonlocal])
        if "u" in self.fields:
            set_mesh_coordinates(self.fields["u"].function_space.mesh, self.fields["u"].x.array, mode="add")
        self.t += h


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

#         self.p_l = dfx.fem.Function(self.QS)
#         self.p_nl_q = get_local(self.p_l).copy()

#         self.p_nl = dfx.fem.Function(function_space)

#         test_function = ufl.TestFunction(function_space)
#         trial_function = ufl.TrialFunction(function_space)
#         b_form = ufl.inner(test_function, self.p_l) * self.quadrature_rule.dx
#         A_form = (
#             ufl.inner(test_function, trial_function)
#             + self.l**2.0
#             * ufl.inner(ufl.grad(test_function), ufl.grad(trial_function))
#         ) * self.quadrature_rule.dx

#         self.problem = dfx.fem.petsc.LinearProblem(A_form, b_form, u=self.p_nl)
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


class CDMNonlocalMechanics(CDMSolver):
    nonlocal_solver: NonlocalInterface
    mechanics_solver: CDM3D
    # fields: dict[str, df.fem.Function]
    # q_fields: dict[str, df.fem.Function | ufl.core.expr.Expr]
    function_space: df.fem.FunctionSpace
    t: float

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        velocity_space: df.fem.FunctionSpace,
        nonlocal_space: df.fem.FunctionSpace,
        t0: float,
        f_ext: Callable | None,
        bcs: list[df.fem.DirichletBCMetaClass],
        rust_model: RustConstitutiveModel,
        parameters: dict[str, float],
        nonlocal_parameters: dict[str, float],
        Q_local: str,
        Q_local_damage: str | None,
        Q_nonlocal: str,
        Q_nonlocal_rate: str,
        quadrature_rule: QuadratureRule,
        mechanics_solver: Type[CDM3D] = CDMPlaneStrain,
        nonlocal_solver: Type[NonlocalInterface] = CDMNonlocal,
        additional_output: list[str] | None = None,
        mass_mechanics: df.fem.Function | None = None,
        mass_nonlocal: df.fem.Function | None = None,
        calculate_bulk_viscosity: bool = False,
        nonlocal_initial_config: bool = True,
    ) -> None:
        mass_mechanics = (
            mass_mechanics
            if mass_mechanics is not None
            else diagonal_mass(velocity_space, parameters["rho"], invert=True)
        )
        mass_nonlocal = (
            mass_nonlocal
            if mass_nonlocal is not None or nonlocal_solver == ImplicitNonlocal
            else diagonal_mass(nonlocal_space, nonlocal_parameters["zeta"], invert=True)
        )
        mechanics_solver = mechanics_solver(
            velocity_space,
            t0,
            f_ext,
            bcs,
            mass_mechanics,
            rust_model,
            quadrature_rule,
            additional_output,
            calculate_bulk_viscosity=calculate_bulk_viscosity,
        )

        displacements = mechanics_solver.fields["u"] if nonlocal_initial_config else None
        nonlocal_solver = nonlocal_solver(
            Q_local,
            Q_nonlocal_rate,
            t0,
            nonlocal_space,
            mass_nonlocal,
            nonlocal_parameters,
            quadrature_rule,
            mechanics_solver.q_fields,
            mechanics_solver.model.input[Q_nonlocal],
            Q_local_damage=Q_local_damage,
            displacements=displacements,
        )

        # add all fields from the solver to this class for easier postprocessing
        fields = mechanics_solver.fields.copy()
        for key, field in nonlocal_solver.fields.items():
            fields[key] = field

        q_fields = mechanics_solver.q_fields.copy()
        for key, field in nonlocal_solver.q_fields.items():
            if key not in q_fields:
                q_fields[key] = field

        super().__init__(
            nonlocal_solver=nonlocal_solver,
            mechanics_solver=mechanics_solver,
            fields=fields,
            q_fields=q_fields,
            function_space=velocity_space,
            t=t0,
        )

    def step(self, h: float) -> None:
        self.mechanics_solver.step(h, self.nonlocal_solver.step)
        self.t = self.mechanics_solver.t
