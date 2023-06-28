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
timer_numpy=TTimer("numpy")
for n in sizes:
    rust_law = comfe.PyLinearElastic3D(E,nu)
    eps = np.random.random(6*n)

    sigma_rust = np.zeros(6*n)
    sigma_numpy = np.zeros(6*n)
    Ct = np.zeros(36*n)

    for j in range(n_timings):
        with timer_rust:
            rust_law.evaluate(0.5, sigma_rust, eps, Ct)

    del sigma_rust, sigma_numpy, Ct, eps

mean_rust = np.mean(np.array(timer_rust.timings).reshape(-1,n_timings), axis=1)

plt.plot(sizes, mean_rust, label="very fast rust model")

plt.legend()
plt.show()