import sympy as sp

x, t, eta, xi = sp.symbols('x t eta xi', real=True)

u = eta**2 + 2*eta*xi

# Calculate nonlinear term I1
print("Calculating I1 (nonlinear term)...")

u_squared = u**2 
integrand1 = (x + t + eta + xi) * u_squared

inner_integral1 = sp.integrate(integrand1, (eta, xi/10, x))
print(f"Inner result: {inner_integral1}")

I1 = sp.integrate(inner_integral1, (xi, 0, t))
print(f"I1 = {I1}\n")

# Calculate linear term I2
print("Calculating I2 (linear term)...")

integrand2 = (x*t + eta*xi**2) * u

inner_integral2 = sp.integrate(integrand2, (eta, xi/10, sp.exp(xi)/5))
print(f"Inner result: {inner_integral2}")

I2 = sp.integrate(inner_integral2, (xi, 0, 1))
print(f"I2 = {I2}\n")

# Calculate f(x, t)
print("Calculating f(x, t)...")
u_xt = x**2 + 2*x*t
f = u_xt + I1 + I2

# Simplify and expand
f_simplified = sp.simplify(f)
print(f"f(x, t) = {f_simplified}")

print("\nExpanded:")
f_expanded = sp.expand(f_simplified)
print(f"f(x, t) = {f_expanded}")