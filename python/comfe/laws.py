from abc import ABC, abstractmethod

import dolfinx as df
import numpy as np
from pydantic import BaseModel

from .comfe import PyGradientJH23D, PyJH23D, PyLinearElastic3D, PyLinElas3D
from .helpers import QuadratureRule

__all__ = [
    "RustConstitutiveModel",
    "QuadratureModel",
    "ConstitutiveModel",
    "PyLinElas3D",
    "PyJH23D",
    "PyGradientJH23D",
    "PyLinearElastic3D",
]
RustConstitutiveModel = PyLinElas3D | PyJH23D | PyLinearElastic3D | PyGradientJH23D


class QuadratureModel(ABC):
    @abstractmethod
    def evaluate(self, del_t: float = 1.0) -> None:
        pass

    @abstractmethod
    def evaluate_some(self, del_t: float = 1.0) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass


class ConstitutiveModel(BaseModel):
    rs_object: RustConstitutiveModel
    input: dict[str, df.fem.Function]
    output: dict[str, df.fem.Function]
    ips: np.ndarray[np.uint64] | None = None
    spaces: dict[int | tuple[int, int], df.fem.FunctionSpace]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model: RustConstitutiveModel,
        rule: QuadratureRule,
        mesh: df.mesh.Mesh,
        ips: np.ndarray[np.uint64] | None = None,
        additional_variables: list[str] | None = None,
    ) -> None:
        input, output, spaces = ceate_input_and_output(model, rule, mesh, None, additional_variables)
        super().__init__(rs_object=model, input=input, output=output, ips=ips, spaces=spaces)

    def evaluate(self, del_t=1.0) -> None:
        input = {key: value.vector.array for key, value in self.input.items()}
        output = {key: value.vector.array for key, value in self.output.items()}
        self.rs_object.evaluate(del_t, input, output)

    def evaluate_some(self, del_t=1.0) -> None:
        input = {key: value.vector.array for key, value in self.input.items()}
        output = {key: value.vector.array for key, value in self.output.items()}
        self.rs_object.evaluate_some(del_t, self.ips, input, output)

    def update(self) -> None:
        """
        Writes the updated variables in output to the input, making them the new initial state.
        """
        keys = set(self.input.keys()).intersection(set(self.output.keys()))
        for key in keys:
            self.input[key].vector.array[:] = self.output[key].vector.array

    def __getitem__(self, key: str) -> np.ndarray:
        return self.output[key]


def ceate_input_and_output(
    model: RustConstitutiveModel,
    rule: QuadratureRule,
    mesh: df.mesh.Mesh,
    spaces: dict[int | tuple[int, int], df.fem.FunctionSpace] | None = None,
    optional_variables: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int | tuple[int, int], df.fem.FunctionSpace],]:
    inputs = model.define_input()
    outputs = model.define_output()
    inputs.update(model.define_history())
    outputs.update(model.define_history())

    if optional_variables is None:
        optional_output = {}
        optional_history = {}
    else:
        optional_output = dict(
            filter(
                lambda item: item[0] in optional_variables,
                model.define_optional_output().items(),
            )
        )
        optional_history = dict(
            filter(
                lambda item: item[0] in optional_variables,
                model.define_optional_history().items(),
            )
        )

    outputs.update(optional_output)
    outputs.update(optional_history)
    inputs.update(optional_history)

    spaces = {} if spaces is None else spaces
    input_dict, spaces = _spaces_from_dict(inputs, rule, mesh, spaces)
    output_dict, spaces = _spaces_from_dict(outputs, rule, mesh, spaces)

    return input_dict, output_dict, spaces


def _spaces_from_dict(
    definition: dict[str, int | tuple[int, int]],
    rule: QuadratureRule,
    mesh: df.mesh.Mesh,
    spaces: dict[int | tuple[int, int], df.fem.FunctionSpace],
) -> tuple[dict[str, np.ndarray], dict[int | tuple[int, int], df.fem.FunctionSpace]]:
    q_values_dict = {}
    for key, value in definition.items():
        if isinstance(value, tuple):
            if key not in spaces:
                spaces[value] = rule.create_quadrature_tensor_space(mesh, value)
            q_values_dict[key] = df.fem.Function(spaces[value], name=key)
        elif isinstance(value, int) and value > 1:
            if key not in spaces:
                spaces[value] = rule.create_quadrature_vector_space(mesh, value)
            q_values_dict[key] = df.fem.Function(spaces[value], name=key)
        elif value == 1:
            if key not in spaces:
                spaces[value] = rule.create_quadrature_space(mesh)
            q_values_dict[key] = df.fem.Function(spaces[value], name=key)

    return q_values_dict, spaces
