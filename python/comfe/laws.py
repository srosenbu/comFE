from .comfe import PyLinElas3D, PyJH23D, PyLinearElastic3D
import dolfinx as df
from pydantic import BaseModel
import numpy as np
from .helpers import QuadratureRule
from abc import abstractmethod, ABC

RustConstitutiveModel = PyLinElas3D | PyJH23D | PyLinearElastic3D

class QuadratureModel(ABC):
    @abstractmethod
    def evaluate(self, del_t:float = 1.) -> None:
        pass
    @abstractmethod
    def evaluate_some(self, del_t:float = 1.) -> None:
        pass
    @abstractmethod
    def update(self) -> None:
        pass

class ConstitutiveModel(BaseModel):
    _rs_object: RustConstitutiveModel
    _input: dict[str, df.fem.Function]
    _output: dict[str, df.fem.Function]
    _ips: np.ndarray[np.uint64] | None = None
    _spaces: dict[int | tuple[int,int], df.fem.FunctionSpace]

    def __init__(
            self, 
            model: RustConstitutiveModel, 
            rule: QuadratureRule, 
            mesh: df.mesh.Mesh, 
            ips: np.ndarray[np.uint64] |None = None,
            additional_output: list[str] | None = None,
        ) -> None:
        
        input, output, spaces = ceate_input_and_output(model, rule, mesh, None, additional_output)
        super().__init__(_rs_object=model, _input=input, _output=output, _ips=ips, _spaces=spaces)
    
    def evaluate(self, del_t = 1.) -> None:
        input = {key: value.vector.array for key, value in self._input.items()}
        output = {key: value.vector.array for key, value in self._output.items()}
        self._rs_object.evaluate(del_t, input, output)
    
    def evaluate_some(self, del_t = 1.) -> None:
        input = {key: value.vector.array for key, value in self._input.items()}
        output = {key: value.vector.array for key, value in self._output.items()}
        self._rs_object.evaluate_some(del_t, self._ips, input, output)
    
    def update(self) -> None:
        for key, function in self._input.items():
            function.vector.array[:] = self._output[key].vector.array

    def __getitem__(self, key: str) -> np.ndarray:
        return self._output[key]
    


def ceate_input_and_output(
        model: RustConstitutiveModel,
        rule: QuadratureRule,
        mesh:df.mesh.Mesh,
        spaces:dict[int | tuple[int,int], df.fem.FunctionSpace] | None = None,
        optional_output: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int | tuple[int,int], df.fem.FunctionSpace]]:
    inputs = model.define_input()
    outputs = model.define_output()
    optional_outputs = dict(filter(lambda item:item[0] in optional_output, model.define_optional_output().items()))
    spaces = {} if spaces is None else spaces
    input_dict, spaces = _spaces_from_dict(inputs, rule, mesh, spaces)
    output_dict, spaces= _spaces_from_dict(outputs, rule, mesh, spaces)
    optional_output_dict, spaces  = _spaces_from_dict(optional_outputs, rule, mesh, spaces)
    output_dict.update(optional_output_dict)
    return input_dict, output_dict, spaces

    

def _spaces_from_dict(
        definition: dict[str, int | tuple[int,int]],
        rule: QuadratureRule,
        mesh:df.mesh.Mesh,
        spaces:dict[int | tuple[int,int], df.fem.FunctionSpace],
    ) -> tuple[dict[str, np.ndarray], dict[int | tuple[int,int], df.fem.FunctionSpace]]:
    
    q_values_dict = {}
    for key, value in definition.items():
        if isinstance(value, tuple):
            if key not in spaces:
                spaces[key] = rule.create_quadrature_tensor_space(mesh, value)
            q_values_dict[key] = df.fem.Function(spaces[key], name=key)
        elif isinstance(value, int) and value > 1:
            if key not in spaces:
                spaces[key] = rule.create_quadrature_vector_space(mesh, value)
            q_values_dict[key] = df.fem.Function(spaces[key], name=key)
        elif value == 1:
            if key not in spaces:
                spaces[key] = rule.create_quadrature_space(mesh)
            q_values_dict[key] = df.fem.Function(spaces[key], name=key)

    return q_values_dict, spaces

