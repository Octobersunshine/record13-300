import numpy as np
from typing import Callable, Optional, Tuple


class TrapezoidalIntegrator:
    def __init__(self, n_intervals: int = 1000):
        if n_intervals < 1:
            raise ValueError("n_intervals must be at least 1")
        self.n_intervals = n_intervals

    def integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n_intervals: Optional[int] = None,
    ) -> float:
        if a > b:
            a, b = b, a
            sign = -1.0
        else:
            sign = 1.0

        if a == b:
            return 0.0

        n = n_intervals if n_intervals is not None else self.n_intervals
        x = np.linspace(a, b, n + 1)
        h = (b - a) / n
        y = f(x)
        result = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
        return sign * result

    def integrate_points(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-dimensional arrays")
        if x.size != y.size:
            raise ValueError("x and y must have the same length")
        if x.size < 2:
            raise ValueError("at least 2 points are required")

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        dx = np.diff(x_sorted)
        if np.any(dx <= 0):
            raise ValueError("x values must be strictly increasing after sorting")

        return float(np.sum(0.5 * (y_sorted[:-1] + y_sorted[1:]) * dx))

    def integrate_adaptive(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        tol: float = 1e-8,
        max_iter: int = 20,
    ) -> Tuple[float, int]:
        if a > b:
            a, b = b, a
            sign = -1.0
        else:
            sign = 1.0

        if a == b:
            return 0.0, 0

        n = 1
        prev_result = self.integrate(f, a, b, n)

        for iteration in range(1, max_iter + 1):
            n *= 2
            current_result = self.integrate(f, a, b, n)
            error = abs(current_result - prev_result) / 3.0

            if error < tol:
                return sign * current_result, iteration

            prev_result = current_result

        return sign * prev_result, max_iter


def trapezoidal(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_intervals: int = 1000,
) -> float:
    integrator = TrapezoidalIntegrator(n_intervals)
    return integrator.integrate(f, a, b)
