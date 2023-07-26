import comfe
import numpy as np

stress = np.zeros(200*6)
strain = np.ones(200*6)
tangents = np.zeros(200*36)
ips = np.arange(100, dtype=np.uint64)
model=comfe.PyLinearElastic3D({"E":42.,"nu": 0.3})
#model.evaluate(0.5, stress[6:], strain[6:], tangents[36:])
print(tangents[:36].reshape(6,6))
print(tangents[36:72].reshape(6,6))
print(stress)

model.evaluate_some(0.5, stress, strain, tangents, ips)
print(np.linalg.norm(tangents[:100*36]), np.linalg.norm(tangents[100*36:]))
# model.evaluate(0.5, stress, stress, tangents)

#model2=comfe.PyLinElas3D({"E":42.,"nu": 0.3})

#model2.evaluate(0.5, stress, strain, tangents)
#print(model2)
