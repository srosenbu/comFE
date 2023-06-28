import comfe
import numpy as np
stress = np.zeros(2*6)
strain = np.ones(2*6)
model=comfe.PyLinearElastic3D(42.,0.3)
model.evaluate(0.5, stress, strain)

print(stress)

