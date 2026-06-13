import numpy as np
import math
from numerical_integration import TrapezoidalIntegrator, trapezoidal


def test_fixed_vs_auto():
    print("=" * 70)
    print("【修复验证】区间步长过大 Bug 修复前后对比")
    print("=" * 70)

    test_cases = [
        ("∫[0,1] x² dx", lambda x: x ** 2, 0, 1, 1.0 / 3.0),
        ("∫[0,π] sin(x) dx", np.sin, 0, np.pi, 2.0),
        ("∫[0,2] x³ dx", lambda x: x ** 3, 0, 2, 4.0),
        ("∫[0,10] cos(x) dx", np.cos, 0, 10, np.sin(10)),
        ("∫[-1,1] e^(-x²) dx", lambda x: np.exp(-x ** 2), -1, 1, np.sqrt(np.pi) * math.erf(1)),
        ("∫[0,4] 1/(1+x²) dx", lambda x: 1.0 / (1 + x ** 2), 0, 4, np.arctan(4)),
    ]

    small_n_values = [1, 2, 5, 10, 20]

    for name, f, a, b, exact in test_cases:
        print(f"\n{'-' * 70}")
        print(f"测试函数: {name}")
        print(f"精确值: {exact:.12f}")
        print(f"{'n':>6s} | {'固定步长结果':>16s} | {'固定误差':>12s} | {'自动细化结果':>16s} | {'自动误差':>12s}")
        print("-" * 70)

        for n in small_n_values:
            fixed_result = trapezoidal(f, a, b, n_intervals=n, auto_refine=False)
            auto_result = trapezoidal(f, a, b, n_intervals=n, auto_refine=True)

            fixed_err = abs(fixed_result - exact)
            auto_err = abs(auto_result - exact)

            marker = " <<< 步长过大" if fixed_err > 1e-3 else ""
            print(
                f"{n:>6d} | {fixed_result:>16.10f} | {fixed_err:>12.2e} | "
                f"{auto_result:>16.10f} | {auto_err:>12.2e}{marker}"
            )

    print(f"\n{'=' * 70}")
    print("【边界条件测试】")
    print("=" * 70)

    print("\n1. 积分上下限相等 (a = b = 5):")
    result = trapezoidal(np.sin, 5, 5)
    print(f"   结果: {result:.10f} (应为 0.0)")

    print("\n2. 上下限颠倒 (∫[π,0] sin(x) dx):")
    r1 = trapezoidal(np.sin, 0, np.pi, n_intervals=5)
    r2 = trapezoidal(np.sin, np.pi, 0, n_intervals=5)
    print(f"   ∫[0,π] = {r1:.10f},  ∫[π,0] = {r2:.10f}")
    print(f"   两者之和: {r1 + r2:.10f} (应为 0.0)")

    print("\n3. 极大区间 (∫[0,1000] sin(x)/x dx 从很小开始):")
    f_sinc = lambda x: np.where(x == 0, 1.0, np.sin(x) / x)
    result = trapezoidal(f_sinc, 0.001, 10, n_intervals=2, auto_refine=True, tol=1e-6)
    expected = np.array([1.658346])
    print(f"   自动细化结果: {result:.6f}")

    print("\n" + "=" * 70)
    print("【离散点数据欠采样警告测试】")
    print("=" * 70)
    print("\n使用 6 个稀疏点积分 ∫[0,π] sin(x) dx (期望 ~2.0):")
    import warnings
    integrator = TrapezoidalIntegrator(tol=1e-3)
    x_sparse = np.linspace(0, np.pi, 6)
    y_sparse = np.sin(x_sparse)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = integrator.integrate_points(x_sparse, y_sparse)
        if w:
            print(f"   警告: {w[0].message}")
        print(f"   积分结果: {r:.10f}, 误差: {abs(r - 2.0):.2e}")

    print("\n使用 100 个稠密点 (无警告):")
    x_dense = np.linspace(0, np.pi, 100)
    y_dense = np.sin(x_dense)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = integrator.integrate_points(x_dense, y_dense)
        if w:
            print(f"   警告: {w[0].message}")
        else:
            print(f"   (无警告，数据密度足够)")
        print(f"   积分结果: {r:.10f}, 误差: {abs(r - 2.0):.2e}")

    print("\n✅ 所有测试完成")


if __name__ == "__main__":
    test_fixed_vs_auto()
