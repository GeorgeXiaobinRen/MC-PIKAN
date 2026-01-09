import sympy as sp

# 定义符号变量
x, t, eta, xi = sp.symbols('x t eta xi', real=True)

# 已知精确解 u(eta, xi) = eta^2 + 2*eta*xi
u = 1

# ========== 计算第二项 I1 ==========
print("计算第二项 I1 (非线性积分项)...")

# 被积函数: (x+t+eta+xi) * u^2
u_squared = u**2  # (eta^2 + 2*eta*xi)^2
integrand1 = (x + t + eta + xi) * u_squared

# 积分顺序: 先对 eta 积分，再对 xi 积分
# 内层积分: ∫_{xi/10}^{x} ... dη
inner_integral1 = sp.integrate(integrand1, (eta, xi/10, x))
print(f"内层积分结果: {inner_integral1}")

# 外层积分: ∫_{0}^{t} ... dξ
I1 = sp.integrate(inner_integral1, (xi, 0, t))
print(f"I1 = {I1}")
print()

# ========== 计算第三项 I2 ==========
print("计算第三项 I2 (线性积分项)...")

# 被积函数: (x*t + eta*xi^2) * u
integrand2 = (x*t + eta*xi**2) * u

# 积分区域 Ω: 0 < ξ < 1, ξ/10 < η < exp(ξ)/5
# 内层积分: ∫_{xi/10}^{exp(xi)/5} ... dη
inner_integral2 = sp.integrate(integrand2, (eta, xi/10, sp.exp(xi)/5))
print(f"内层积分结果: {inner_integral2}")

# 外层积分: ∫_{0}^{1} ... dξ
I2 = sp.integrate(inner_integral2, (xi, 0, 1))
print(f"I2 = {I2}")
print()

# ========== 计算 f(x, t) ==========
print("计算 f(x, t)...")
u_xt = u
f = u_xt + I1 + I2

# 简化表达式
f_simplified = sp.simplify(f)
print(f"f(x, t) = {f_simplified}")

# 也可以展开看看
print("\n展开形式:")
f_expanded = sp.expand(f_simplified)
print(f"f(x, t) = {f_expanded}")