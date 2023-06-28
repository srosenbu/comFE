import comfe
import numpy as np
stress = np.zeros(2000000*6)
strain = np.ones(2000000*6)
tangents = np.zeros(2000000*36)
model=comfe.PyLinearElastic3D(42.,0.3)
model.evaluate(0.5, stress, strain, tangents)
print(tangents[:36].reshape(6,6))
print(stress)
model.evaluate(0.5, stress, stress, tangents)
