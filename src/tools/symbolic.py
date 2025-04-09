import sympy as sp

x, y, L, ax, ay, bx, by, A, B = sp.symbols('x y L ax ay bx by A B')
u = sp.sin(x)*sp.sin(y)
kappa = A*sp.exp(-((x-ax)**2 + (y-ay)**2)) + 1
eta = B*sp.exp(-((x-bx)**2 + (y-by)**2)) + 1
f = -sp.diff(kappa*sp.diff(u, x), x) - sp.diff(kappa*sp.diff(u, y), y) + eta**2*u
#print(f)

print(-sp.diff(sp.diff(u, x), x) - sp.diff(sp.diff(u, y), y))