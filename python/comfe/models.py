from comfe_py import PyDruckerPrager3D, PyLinearElasticity3D, PyMisesPlasticity3D
import numpy as np
from fenics_constitutive import IncrSmallStrainModel, StressStrainConstraint


__all__ = ["DruckerPrager3D", "MisesPlasticity3D", "LinearElasticity3D"]


def fenics_constitutive_wrapper(rust_model):
    def decorator(cls):
        # Overwrite __init__
        def __init__(self, parameters: np.ndarray) -> None:
            self.model = rust_model(parameters)

        cls.__init__ = __init__

        # Add evaluate method
        def evaluate(
            self,
            t: float,
            del_t: float,
            grad_del_u: np.ndarray,
            stress: np.ndarray,
            tangent: np.ndarray,
            history: dict[str, np.ndarray] | None,
        ) -> None:
            # Your implementation here
            self.model.evaluate(
                t,
                del_t,
                grad_del_u,
                stress,
                tangent,
                history,
            )

        cls.evaluate = evaluate

        # Add constraint property
        def constraint(self) -> StressStrainConstraint:
            python_constraint = StressStrainConstraint[
                str(self.model.constraint).split(".")[-1]
            ]
            assert (
                python_constraint.stress_strain_dim
                == self.model.constraint.stress_strain_dim
                and python_constraint.geometric_dim == self.model.geometric_dim
            )
            return StressStrainConstraint[str(self.model.constraint).split(".")[-1]]

        cls.constraint = property(constraint)

        # Add history_dim property
        def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
            # Your implementation here
            return self.model.history_dim

        cls.history_dim = property(history_dim)
        cls.__abstractmethods__ = frozenset()
        return cls

    return decorator


@fenics_constitutive_wrapper(PyLinearElasticity3D)
class LinearElasticity3D(IncrSmallStrainModel):
    pass


@fenics_constitutive_wrapper(PyDruckerPrager3D)
class DruckerPrager3D(IncrSmallStrainModel):
    pass


@fenics_constitutive_wrapper(PyMisesPlasticity3D)
class MisesPlasticity3D(IncrSmallStrainModel):
    pass
