import numpy as np
import time
import matplotlib.pyplot as plt
import comfe

E=42.
nu=0.3
m = 20
sizes = [2**i for i in range(m)]
n_timings = 100

class TTimer:
    def __init__(self, what=""):
        self.what = what
        self.timings = []

    def evaluation(self):
        vec = np.array(self.timings)
        return {"mean": vec.mean(), "std": vec.std(), "measurements": vec.size}

    def total(self):
        return np.sum(self.timings)

    def mean(self):
        return np.mean(self.timings)

    def std(self):
        return np.std(self.timings)

    def to_array(self):
        return np.array(self.timings)

    def __str__(self):
        dic = self.evaluation()
        return f"Timer of {self.what},mean:{dic['mean']}\nstd:{dic['std']}\n#measurements:{dic['measurements']}"

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # df.MPI.barrier(df.MPI.comm_world)
        ms = (time.perf_counter() - self.start) * 1000
        self.timings.append(ms)
timer_rust=TTimer("very fast rust model")
timer_rust_box=TTimer("very fast rust model with Box<>")
timer_indices=TTimer("rust model with indices")
for n in sizes:
    rust_law = comfe.PyLinearElastic3D({"E":E,"nu":nu})
    rust_law_box = comfe.py_new_linear_elastic_3d({"E":E,"nu":nu})
    rust_law_indices = comfe.PyLinearElastic3D({"E":E,"nu":nu})
    eps = np.random.random(6*n)

    sigma_rust = np.zeros(6*n)
    sigma_rust_box = np.zeros(6*n)
    sigma_indices = np.zeros(6*n)
    if n >= 4:
        # we assume that there are always at least 4 quadrature points
        # therefore, the ips array is locally ordered
        # It gets a little faster under this assumption, but ordered ips are better
        indices = np.arange(n, dtype=np.uint64).reshape(-1,4)
        permutation_indices = np.random.permutation(n//4).astype(np.uint64)
        indices = indices[permutation_indices].reshape(-1)
    else:
        indices = np.arange(n, dtype=np.uint64)
 
    Ct = np.zeros(36*n)
    input = {"mandel_strain": eps}
    output = {"mandel_stress": sigma_rust_box, "mandel_tangent": Ct}
    #print(rust_law_box.define_input(), rust_law_box.define_output())
    for j in range(n_timings):
        with timer_rust_box:
            rust_law_box.evaluate(0.5, input, output)
    for j in range(n_timings):
        with timer_rust:
            rust_law.evaluate(0.5, sigma_rust, eps, Ct)
    for j in range(n_timings):
        with timer_indices:
            rust_law_indices.evaluate_some(0.5, sigma_indices, eps, Ct, indices)

    del sigma_rust, sigma_rust_box, sigma_indices, Ct, eps

mean_rust = np.mean(np.array(timer_rust.timings).reshape(-1,n_timings), axis=1)
mean_rust_box = np.mean(np.array(timer_rust_box.timings).reshape(-1,n_timings), axis=1)
mean_indices = np.mean(np.array(timer_indices.timings).reshape(-1,n_timings), axis=1)
plt.plot(sizes, mean_rust, label="very fast rust model")
plt.plot(sizes, mean_rust_box, label="very fast rust model with Box<>")
plt.plot(sizes, mean_indices, label="rust model with indices")

plt.legend()
plt.show()