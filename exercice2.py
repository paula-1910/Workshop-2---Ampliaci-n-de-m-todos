import numpy as np
import matplotlib.pyplot as plt

# 1. Definición de la función pendiente f(t, y) y la solución exacta
def f(t, y):
    return (t / 2.0) * (2.0 - y)

def y_exact(t):
    return 2.0 - np.exp(- (t**2) / 4.0)

# 2. Configuración del problema
t0, t_final = 0.0, 4.0
y0 = 1.0
h_values = [0.1, 0.01, 0.001]

# 3. Crear figura para la gráfica
plt.figure(figsize=(10, 6))

# Dibujar la solución exacta
t_exact = np.linspace(t0, t_final, 1000)
plt.plot(t_exact, y_exact(t_exact), 'k-', linewidth=2, label='Solución Exacta: $y(t) = 2 - e^{-t^2/4}$')

# 4. Bucle para resolver con cada tamaño de paso h
colors = ['r--', 'g--', 'b:']
for h, color in zip(h_values, colors):
    N = int((t_final - t0) / h)
    t = np.linspace(t0, t_final, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    
    # Bucle del Trapecio Implícito
    for n in range(N):
        tn = t[n]
        tn1 = t[n+1]
        yn = y[n]
        fn = f(tn, yn)
        
        # Fórmula despejada del Trapecio Implícito
        y[n+1] = (yn + (h / 2.0) * fn + (h * tn1) / 2.0) / (1.0 + (h * tn1) / 4.0)
        
    # Graficar el resultado numérico
    plt.plot(t, y, color, label=f'Trapecio (h = {h})')

# 5. Formato y etiquetas de la gráfica
plt.title('Comparación del Método del Trapecio para distintos tamaños de paso $h$', fontsize=12)
plt.xlabel('Tiempo ($t$)', fontsize=11)
plt.ylabel('$y(t)$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.show()