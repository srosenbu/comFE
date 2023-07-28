from .comfe import PyLinElas3D, PyJH23D, PyLinearElastic3D
import dolfinx as df
from pydantic import BaseModel
import numpy as np
from .helpers import QuadratureRule
from abc import abstractmethod, ABC

RustConstitutiveModel = PyLinElas3D | PyJH23D | PyLinearElastic3D


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
        additional_output: list[str] | None = None,
    ) -> None:
        input, output, spaces = ceate_input_and_output(
            model, rule, mesh, None, additional_output
        )
        super().__init__(
            rs_object=model, input=input, output=output, ips=ips, spaces=spaces
        )

    def evaluate(self, del_t=1.0) -> None:
        input = {key: value.vector.array for key, value in self.input.items()}
        output = {key: value.vector.array for key, value in self.output.items()}
        self.rs_object.evaluate(del_t, input, output)

    def evaluate_some(self, del_t=1.0) -> None:
        input = {key: value.vector.array for key, value in self.input.items()}
        output = {key: value.vector.array for key, value in self.output.items()}
        self.rs_object.evaluate_some(del_t, self.ips, input, output)

    def update(self) -> None:
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
    optional_output: list[str] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[int | tuple[int, int], df.fem.FunctionSpace],
]:
    inputs = model.define_input()
    outputs = model.define_output()
    if optional_output is None:
        optional_output = {}
    else:
        optional_output = dict(
            filter(
                lambda item: item[0] in optional_output,
                model.define_optional_output().items(),
            )
        )
    spaces = {} if spaces is None else spaces
    input_dict, spaces = _spaces_from_dict(inputs, rule, mesh, spaces)
    output_dict, spaces = _spaces_from_dict(outputs, rule, mesh, spaces)
    optional_output_dict, spaces = _spaces_from_dict(
        optional_output, rule, mesh, spaces
    )
    output_dict.update(optional_output_dict)
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
