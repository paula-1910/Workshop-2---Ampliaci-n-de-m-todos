import numpy as np
import matplotlib.pyplot as plt

# EXERCISE 4: NUMERICAL SOLUTION USING 4TH-ORDER RUNGE-KUTTA METHOD (RK4)

#
# RESULTS ANALYSIS AND COMPARISON WITH EXERCISE 2 (TRAPEZOIDAL METHOD):
#
# 1. Higher Precision of RK4:
#    Since RK4 is a 4th-order method O(h^4), it achieves extremely high precision 
#    even with a very large step size like h = 1. In Exercise 2, the Trapezoidal 
#    method O(h^2) required smaller step sizes (h <= 0.1) to achieve a similar 
#    level of agreement with the exact solution.
#
# 2. Comparison for Large Step Sizes (h = 1 vs h = 0.1):
#    - For h = 1, RK4 already tracks the true analytical solution y(t) = 2 - exp(-t^2/4) 
#      with remarkable accuracy.
#    - For h = 0.1 and h = 0.01, the RK4 solution becomes virtually identical to 
#      the exact solution, yielding near-zero machine precision errors.
#
# 3. Efficiency (Computational Cost):
#    Although each RK4 step requires 4 function evaluations (k1, k2, k3, k4), 
#    its O(h^4) convergence rate allows using much larger step sizes h to obtain 
#    the same or better accuracy compared to 2nd-order methods like Trapezoidal.


# 1. Define the ODE right-hand side function f(t, y) and exact solution
def f(t, y):
    return (t / 2.0) * (2.0 - y)

def y_exact(t):
    return 2.0 - np.exp(- (t**2) / 4.0)

# 2. Problem parameters
t0, t_final = 0.0, 4.0
y0 = 1.0
h_values = [1.0, 0.1, 0.01]

# 3. Create plot figure
plt.figure(figsize=(10, 6))

# Plot analytical exact solution
t_exact = np.linspace(t0, t_final, 1000)
plt.plot(t_exact, y_exact(t_exact), 'k-', linewidth=2, label=r'Exact Solution: $y(t) = 2 - e^{-t^2/4}$')

# 4. Loop over each step size h using Classical RK4
colors = ['m--o', 'r--', 'g:']
for h, color in zip(h_values, colors):
    N = int((t_final - t0) / h)
    t = np.linspace(t0, t_final, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    
    # RK4 Integration Loop
    for n in range(N):
        tn = t[n]
        yn = y[n]
        
        # Calculate RK4 slopes
        k1 = f(tn, yn)
        k2 = f(tn + 0.5 * h, yn + 0.5 * h * k1)
        k3 = f(tn + 0.5 * h, yn + 0.5 * h * k2)
        k4 = f(tn + h, yn + h * k3)
        
        # Advance solution
        y[n+1] = yn + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    # Plot numerical result
    if h == 1.0:
        plt.plot(t, y, color, markersize=5, label=f'RK4 (h = {h})')
    else:
        plt.plot(t, y, color, label=f'RK4 (h = {h})')

# 5. Graph formatting
plt.title('Comparison of 4th-Order Runge-Kutta (RK4) for Different Step Sizes $h$', fontsize=12)
plt.xlabel('Time ($t$)', fontsize=11)
plt.ylabel('$y(t)$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()