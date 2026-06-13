import numpy as np
from typing import Callable, Optional, Tuple


class TrapezoidalIntegrator:
    def __init__(
        self,
        n_intervals: int = 1000,
        auto_refine: bool = True,
        tol: float = 1e-8,
        max_refine_iter: int = 20,
        method: str = "trapezoidal",
    ):
        if n_intervals < 1:
            raise ValueError("n_intervals must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_refine_iter < 1:
            raise ValueError("max_refine_iter must be at least 1")
        method = method.lower()
        if method not in ("trapezoidal", "simpson"):
            raise ValueError("method must be 'trapezoidal' or 'simpson'")

        self.n_intervals = n_intervals
        self.auto_refine = auto_refine
        self.tol = tol
        self.max_refine_iter = max_refine_iter
        self.method = method

    def _trapz_fixed(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n: int,
    ) -> float:
        x = np.linspace(a, b, n + 1)
        h = (b - a) / n
        y = f(x)
        return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

    def _simpson_fixed(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n: int,
    ) -> float:
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n + 1)
        h = (b - a) / n
        y = f(x)
        return (h / 3.0) * (
            y[0]
            + y[-1]
            + 4.0 * np.sum(y[1:-1:2])
            + 2.0 * np.sum(y[2:-2:2])
        )

    def _fixed_integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n: int,
        method: str,
    ) -> float:
        if method == "simpson":
            return self._simpson_fixed(f, a, b, n)
        else:
            return self._trapz_fixed(f, a, b, n)

    def integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n_intervals: Optional[int] = None,
        auto_refine: Optional[bool] = None,
        tol: Optional[float] = None,
        max_refine_iter: Optional[int] = None,
        method: Optional[str] = None,
    ) -> float:
        if a > b:
            a, b = b, a
            sign = -1.0
        else:
            sign = 1.0

        if a == b:
            return 0.0

        n = n_intervals if n_intervals is not None else self.n_intervals
        use_refine = auto_refine if auto_refine is not None else self.auto_refine
        rel_tol = tol if tol is not None else self.tol
        max_iter = (
            max_refine_iter if max_refine_iter is not None else self.max_refine_iter
        )
        _method = method.lower() if method is not None else self.method
        if _method not in ("trapezoidal", "simpson"):
            raise ValueError("method must be 'trapezoidal' or 'simpson'")

        result = self._fixed_integrate(f, a, b, n, _method)

        if not use_refine:
            return sign * result

        error_divisor = 15.0 if _method == "simpson" else 3.0

        prev_result = result
        current_n = n
        for _ in range(max_iter):
            current_n *= 2
            current_result = self._fixed_integrate(f, a, b, current_n, _method)
            error_estimate = abs(current_result - prev_result) / error_divisor

            scale = max(abs(b - a), 1.0)
            if error_estimate < rel_tol * scale:
                return sign * current_result

            prev_result = current_result

        return sign * prev_result

    def integrate_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: Optional[str] = None,
    ) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-dimensional arrays")
        if x.size != y.size:
            raise ValueError("x and y must have the same length")
        if x.size < 2:
            raise ValueError("at least 2 points are required")

        _method = method.lower() if method is not None else self.method
        if _method not in ("trapezoidal", "simpson"):
            raise ValueError("method must be 'trapezoidal' or 'simpson'")

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        dx = np.diff(x_sorted)
        if np.any(dx <= 0):
            raise ValueError("x values must be strictly increasing after sorting")

        is_uniform = np.allclose(dx, dx[0])

        if _method == "simpson" and is_uniform and x_sorted.size >= 3:
            n = x_sorted.size - 1
            if n % 2 != 0:
                import warnings
                warnings.warn(
                    "Simpson's rule requires an even number of intervals. "
                    "Using trapezoidal for the last sub-interval.",
                    UserWarning,
                    stacklevel=2,
                )
                n_even = n - 1
                h = dx[0]
                simps_result = (h / 3.0) * (
                    y_sorted[0]
                    + y_sorted[n_even]
                    + 4.0 * np.sum(y_sorted[1:n_even:2])
                    + 2.0 * np.sum(y_sorted[2:n_even - 1:2])
                )
                trap_last = 0.5 * (y_sorted[n_even] + y_sorted[-1]) * dx[-1]
                result = float(simps_result + trap_last)
            else:
                h = dx[0]
                result = float(
                    (h / 3.0) * (
                        y_sorted[0]
                        + y_sorted[-1]
                        + 4.0 * np.sum(y_sorted[1:-1:2])
                        + 2.0 * np.sum(y_sorted[2:-2:2])
                    )
                )
        else:
            if _method == "simpson" and not is_uniform:
                import warnings
                warnings.warn(
                    "Simpson's rule requires uniformly spaced x values. "
                    "Falling back to trapezoidal rule.",
                    UserWarning,
                    stacklevel=2,
                )
            result = float(np.sum(0.5 * (y_sorted[:-1] + y_sorted[1:]) * dx))

        if _method == "trapezoidal" and 3 < x.size <= 20 and is_uniform:
            coarse_x = x_sorted[::2]
            coarse_y = y_sorted[::2]
            if coarse_x.size >= 2:
                coarse_dx = np.diff(coarse_x)
                coarse_result = float(
                    np.sum(0.5 * (coarse_y[:-1] + coarse_y[1:]) * coarse_dx)
                )
                error_estimate = abs(result - coarse_result) / 3.0
                scale = max(abs(x_sorted[-1] - x_sorted[0]), 1.0)
                if error_estimate >= self.tol * scale:
                    import warnings
                    warnings.warn(
                        f"Discrete data may be under-sampled (only {x.size} points). "
                        f"Estimated error: {error_estimate:.2e} exceeds tolerance "
                        f"{self.tol * scale:.2e}. Consider providing more data points.",
                        UserWarning,
                        stacklevel=2,
                    )

        return result

    def integrate_adaptive(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        tol: float = 1e-8,
        max_iter: int = 20,
        method: Optional[str] = None,
    ) -> Tuple[float, int]:
        if a > b:
            a, b = b, a
            sign = -1.0
        else:
            sign = 1.0

        if a == b:
            return 0.0, 0

        _method = method.lower() if method is not None else self.method
        if _method not in ("trapezoidal", "simpson"):
            raise ValueError("method must be 'trapezoidal' or 'simpson'")

        error_divisor = 15.0 if _method == "simpson" else 3.0

        n = 2 if _method == "simpson" else 1
        prev_result = self._fixed_integrate(f, a, b, n, _method)

        for iteration in range(1, max_iter + 1):
            n *= 2
            current_result = self._fixed_integrate(f, a, b, n, _method)
            error_estimate = abs(current_result - prev_result) / error_divisor

            scale = max(abs(b - a), 1.0)
            if error_estimate < tol * scale:
                return sign * current_result, iteration

            prev_result = current_result

        return sign * prev_result, max_iter


def trapezoidal(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_intervals: int = 1000,
    auto_refine: bool = True,
    tol: float = 1e-8,
    max_refine_iter: int = 20,
) -> float:
    integrator = TrapezoidalIntegrator(
        n_intervals=n_intervals,
        auto_refine=auto_refine,
        tol=tol,
        max_refine_iter=max_refine_iter,
        method="trapezoidal",
    )
    return integrator.integrate(f, a, b)


def simpson(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_intervals: int = 1000,
    auto_refine: bool = True,
    tol: float = 1e-8,
    max_refine_iter: int = 20,
) -> float:
    integrator = TrapezoidalIntegrator(
        n_intervals=n_intervals,
        auto_refine=auto_refine,
        tol=tol,
        max_refine_iter=max_refine_iter,
        method="simpson",
    )
    return integrator.integrate(f, a, b)
