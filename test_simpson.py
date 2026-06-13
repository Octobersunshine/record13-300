import numpy as np
import math
from numerical_integration import TrapezoidalIntegrator, trapezoidal, simpson


def test_simpson_vs_trapezoidal():
    print("=" * 80)
    print("【辛普森法 vs 梯形法】精度对比测试")
    print("=" * 80)

    test_cases = [
        ("∫[0,1] x⁴ dx", lambda x: x ** 4, 0, 1, 1.0 / 5.0),
        ("∫[0,π] sin(x) dx", np.sin, 0, np.pi, 2.0),
        ("∫[0,2] x³ dx", lambda x: x ** 3, 0, 2, 4.0),
        ("∫[0,10] cos(x) dx", np.cos, 0, 10, np.sin(10)),
        ("∫[-1,1] e^(-x²) dx", lambda x: np.exp(-x ** 2), -1, 1, np.sqrt(np.pi) * math.erf(1)),
        ("∫[0,4] 1/(1+x²) dx", lambda x: 1.0 / (1 + x ** 2), 0, 4, np.arctan(4)),
    ]

    n_values = [2, 4, 8, 16, 32, 64, 128]

    for name, f, a, b, exact in test_cases:
        print(f"\n{'─' * 80}")
        print(f"测试函数: {name}")
        print(f"精确值: {exact:.12f}")
        print(f"{'n':>6s} | {'梯形法误差':>14s} | {'辛普森法误差':>14s} | {'精度提升':>12s}")
        print("-" * 80)

        for n in n_values:
            trap_result = trapezoidal(f, a, b, n_intervals=n, auto_refine=False)
            simp_result = simpson(f, a, b, n_intervals=n, auto_refine=False)

            trap_err = abs(trap_result - exact)
            simp_err = abs(simp_result - exact)

            if simp_err > 0:
                ratio = trap_err / simp_err
                ratio_str = f"{ratio:.0f}x"
            else:
                ratio_str = "∞"

            marker = ""
            if n <= 8 and trap_err > 1e-3:
                marker = "  <<< 梯形法步长过大"

            print(
                f"{n:>6d} | {trap_err:>14.2e} | {simp_err:>14.2e} | {ratio_str:>12s}{marker}"
            )

    print(f"\n{'=' * 80}")
    print("【自适应对比】达到相同精度所需迭代次数")
    print("=" * 80)

    integrator = TrapezoidalIntegrator()
    tol_values = [1e-6, 1e-10, 1e-12]

    for tol in tol_values:
        print(f"\n容限 tol = {tol:.0e}:")
        print(f"  {'测试函数':<28s} | {'梯形法迭代':>10s} | {'辛普森迭代':>10s} | {'辛普森更快':>12s}")
        print("  " + "-" * 75)

        for name, f, a, b, exact in test_cases:
            _, trap_iters = integrator.integrate_adaptive(f, a, b, tol=tol, method="trapezoidal")
            _, simp_iters = integrator.integrate_adaptive(f, a, b, tol=tol, method="simpson")

            speedup = 2 ** (trap_iters - simp_iters) if trap_iters >= simp_iters else 1.0 / (2 ** (simp_iters - trap_iters))
            speedup_str = f"{speedup:.0f}x" if speedup >= 1 else f"1/{1/speedup:.0f}x"

            short_name = name.split(" ")[0] if len(name) > 28 else name
            print(f"  {short_name:<28s} | {trap_iters:>10d} | {simp_iters:>10d} | {speedup_str:>12s}")

    print(f"\n{'=' * 80}")
    print("【离散点数据辛普森法测试】")
    print("=" * 80)

    x_dense = np.linspace(0, np.pi, 101)
    y_dense = np.sin(x_dense)

    integrator = TrapezoidalIntegrator()

    print(f"\n100 个区间（均匀采样），∫[0,π] sin(x) dx (精确值 = 2.0):")
    r_trap = integrator.integrate_points(x_dense, y_dense, method="trapezoidal")
    r_simp = integrator.integrate_points(x_dense, y_dense, method="simpson")
    print(f"  梯形法结果: {r_trap:.12f}, 误差: {abs(r_trap - 2.0):.2e}")
    print(f"  辛普森结果: {r_simp:.12f}, 误差: {abs(r_simp - 2.0):.2e}")

    print(f"\n11 个稀疏点（奇数个点，n=10 偶区间）:")
    x_sparse = np.linspace(0, np.pi, 11)
    y_sparse = np.sin(x_sparse)
    r_simp_sparse = integrator.integrate_points(x_sparse, y_sparse, method="simpson")
    print(f"  辛普森结果: {r_simp_sparse:.12f}, 误差: {abs(r_simp_sparse - 2.0):.2e}")

    print(f"\n10 个点（n=9 奇区间，应触发警告）:")
    x_odd = np.linspace(0, np.pi, 10)
    y_odd = np.sin(x_odd)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r_simp_odd = integrator.integrate_points(x_odd, y_odd, method="simpson")
        if w:
            print(f"  警告: {w[0].message}")
        print(f"  辛普森+梯形混合结果: {r_simp_odd:.12f}, 误差: {abs(r_simp_odd - 2.0):.2e}")

    print(f"\n非均匀间距点（辛普森法应回退到梯形法并警告）:")
    x_nonuniform = np.array([0, 0.5, 1.0, 2.0, np.pi])
    y_nonuniform = np.sin(x_nonuniform)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r_nonuniform = integrator.integrate_points(x_nonuniform, y_nonuniform, method="simpson")
        if w:
            print(f"  警告: {w[0].message}")
        print(f"  结果: {r_nonuniform:.12f}")

    print("\n✅ 所有辛普森法测试完成")


if __name__ == "__main__":
    test_simpson_vs_trapezoidal()
