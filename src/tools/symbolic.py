import sympy as sp

x, y, L, ax, ay, bx, by, A, B = sp.symbols('x y L ax ay bx by A B')
u = sp.sin(2*sp.pi*x/L)*sp.sin(2*sp.pi*y/L)
kappa = A*sp.exp(-((x-ax)**2 + (y-ay)**2)) + 1
eta = B*sp.exp(-((x-bx)**2 + (y-by)**2)) + 1
f = -sp.diff(kappa*sp.diff(u, x), x) - sp.diff(kappa*sp.diff(u, y), y) + eta**2*u

print(sp.latex(f.subs({L: 6, A: 3.5, B: 1.5, ax: -1, ay: -1, bx: 2, by: 2})))
print("-------------------------")
print(sp.latex(u.subs({L: 6})))