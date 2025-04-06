import numpy as np


class ExplicitMidpoint:
    """
    A class to implement the Runge-Kutta method for solving ODEs. What is special is that this solver works on a
    list of numpy.ndarray instead of arrays themselves. This is because the solver is intended to be used for
    solving several coupled ODEs at once.
    """

    def __init__(self, f: list[callable], y0: list[np.ndarray], t0: float, h_max: float) -> None:
        self.f = f
        self.y = y0
        self.t = t0
        self.h = h_max

    def step(self, del_t, h=None) -> None:
        if h is None:
            h = self.h
        # k1 = [y_i.copy() for y_i in y]
        for y_i, f_i in zip(self.y, self.f):
            k = y_i + h / 2.0 * f_i(self.t, y_i)
            y_i[:] += h / 2.0 * f_i(self.t + h / 2.0, k)
        self.t += h
