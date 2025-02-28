import sympy as sp

x, y, L, ax, ay, bx, by, A, B = sp.symbols('x y L ax ay bx by A B')
u = sp.sin(2*sp.pi*x/L)*sp.sin(2*sp.pi*y/L)
kappa = A*sp.exp(-((x-ax)**2 + (y-ay)**2)) + 1
eta = B*sp.exp(-((x-bx)**2 + (y-by)**2)) + 1
f = -sp.diff(kappa*sp.diff(u, x), x) - sp.diff(kappa*sp.diff(u, y), y) + eta**2*u
print(f)

# Output:-2*pi*A*(2*ax - 2*x)*exp(-(-ax + x)**2 - (-ay + y)**2)*sin(2*pi*y/L)*cos(2*pi*x/L)/L - 2*pi*A*(2*ay - 2*y)*exp(-(-ax + x)**2 - (-ay + y)**2)*sin(2*pi*x/L)*cos(2*pi*y/L)/L + (B*exp(-(-bx + x)**2 - (-by + y)**2) + 1)**2*sin(2*pi*x/L)*sin(2*pi*y/L) + 8*pi**2*(A*exp(-(-ax + x)**2 - (-ay + y)**2) + 1)*sin(2*pi*x/L)*sin(2*pi*y/L)/L**2