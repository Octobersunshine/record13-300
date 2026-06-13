import numpy as np
import math
from numerical_integration import TrapezoidalIntegrator, trapezoidal


def main():
    print("=== 数值积分服务 - 梯形法示例 ===\n")

    integrator = TrapezoidalIntegrator(n_intervals=10000)

    print("【示例 1】计算 ∫[0,1] x² dx")
    f1 = lambda x: x ** 2
    result1 = integrator.integrate(f1, 0, 1)
    exact1 = 1.0 / 3.0
    print(f"  数值结果: {result1:.10f}")
    print(f"  精确结果: {exact1:.10f}")
    print(f"  误差:     {abs(result1 - exact1):.2e}\n")

    print("【示例 2】计算 ∫[0,π] sin(x) dx")
    f2 = np.sin
    result2 = integrator.integrate(f2, 0, np.pi)
    exact2 = 2.0
    print(f"  数值结果: {result2:.10f}")
    print(f"  精确结果: {exact2:.10f}")
    print(f"  误差:     {abs(result2 - exact2):.2e}\n")

    print("【示例 3】计算 ∫[-1,1] e^(-x²) dx (高斯积分)")
    f3 = lambda x: np.exp(-x ** 2)
    result3 = integrator.integrate(f3, -1, 1)
    exact3 = np.sqrt(np.pi) * math.erf(1)
    print(f"  数值结果: {result3:.10f}")
    print(f"  精确结果: {exact3:.10f}")
    print(f"  误差:     {abs(result3 - exact3):.2e}\n")

    print("【示例 4】自适应积分 - ∫[0,10] cos(x) dx")
    f4 = np.cos
    result4_adaptive, iterations = integrator.integrate_adaptive(f4, 0, 10, tol=1e-10)
    result4_simple = integrator.integrate(f4, 0, 10)
    exact4 = np.sin(10)
    print(f"  自适应结果 ({iterations} 次迭代): {result4_adaptive:.12f}")
    print(f"  普通结果 (10000 区间):         {result4_simple:.12f}")
    print(f"  精确结果:                        {exact4:.12f}")
    print(f"  自适应误差:                      {abs(result4_adaptive - exact4):.2e}")
    print(f"  普通误差:                        {abs(result4_simple - exact4):.2e}\n")

    print("【示例 5】从离散点数据积分")
    x_points = np.linspace(0, np.pi, 100)
    y_points = np.sin(x_points)
    result5 = integrator.integrate_points(x_points, y_points)
    print(f"  数值结果: {result5:.10f}")
    print(f"  精确结果: {2.0:.10f}")
    print(f"  误差:     {abs(result5 - 2.0):.2e}\n")

    print("【示例 6】使用便捷函数 trapezoidal()")
    result6 = trapezoidal(lambda x: x ** 3, 0, 2, n_intervals=5000)
    exact6 = 4.0
    print(f"  ∫[0,2] x³ dx = {result6:.10f}")
    print(f"  精确值:        {exact6:.10f}")
    print(f"  误差:          {abs(result6 - exact6):.2e}\n")

    print("【示例 7】积分上下限顺序颠倒")
    result7_normal = trapezoidal(np.sin, 0, np.pi)
    result7_reverse = trapezoidal(np.sin, np.pi, 0)
    print(f"  ∫[0,π] sin(x) dx = {result7_normal:.10f}")
    print(f"  ∫[π,0] sin(x) dx = {result7_reverse:.10f}")
    print(f"  两者之和 = {result7_normal + result7_reverse:.10f} (应为 0)")


if __name__ == "__main__":
    main()
