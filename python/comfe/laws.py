from comfe.comfe import PyConstitutiveModel, new_linear_elastic_3d, new_jh2_3d
import dolfinx as df
from pydantic import BaseModel
import numpy as np
from .helpers import QuadratureRule

class ModelWrapper(BaseModel):
    _rs_object: PyConstitutiveModel
    _input: dict[str, np.ndarray]
    _output: dict[str, np.ndarray]
    _ips: np.ndarray[np.uint64] | None = None
    _additional_output: list[str] = None
    _rule: QuadratureRule
    _spaces: dict[int | tuple[int,int], df.FunctionSpace]

    def evaluate(self, del_t = 1.) -> None:
        self._rs_object.evaluate(del_t, self._input, self._output)
    
    def evaluate_some(self, del_t = 1.) -> None:
        self._rs_object.evaluate_some(del_t, self._ips, self._input, self._output)
    
    def update(self) -> None:
        for key, array in self._input.items():
            array[:] = self._output[key]

    def __getitem__(self, key: str) -> np.ndarray:
        return self._output[key]
    
    def create_input_and_output(self, rs_object: PyConstitutiveModel,n: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        inputs = rs_object.define_input()
        outputs = rs_object.define_output()
        input_dict = {}
        output_dict = {}
        for key, value in inputs.items():
            
            if isinstance(value, tuple):
                if key not in self._spaces:
                    self._spaces[key] = self.rule.create_tensor_function_space()

                input_dict[key] = np.zeros(n*value[0]*value[1])
            elif isinstance(value, int):
                input_dict[key] = np.zeros(n*value)
        for key, value in outputs.items():
            if isinstance(value, tuple):
                input_dict[key] = np.zeros(n*value[0]*value[1])
            elif isinstance(value, int):
                input_dict[key] = np.zeros(n*value)


class LinearElastic(ModelWrapper):
    def __init__(self, parameters: dict[str, float], n:int) -> None:
        rs_object = new_linear_elastic_3d(parameters)
        super().__init__(_rs_object=rs_object, _input={"strain": np.zeros((n, 6))}, _output={"stress": np.zeros((n, 6))})
