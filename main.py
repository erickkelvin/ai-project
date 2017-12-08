import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import decimal

def f(x, y):
    return ((4 -(2.1*(x**2))) + ((1/3.0)*(x**4)))*(x**2) + (x*y) - 4*(1 - y**2)*(y**2)

def grad(*x):
    return [(2*(x[0]**5) - 8.4*(x[0]**3) + 8*x[0] + x[1]), (x[0] + 16*(x[1]**3) - 8*x[1])]

# PLOTS #######################################################################
x = np.linspace(-2.2, 2.2, 100)
y = np.linspace(-1.2, 1.2, 100)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig1 = plt.figure(1, figsize=(8,6))
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
plot_3d = ax1.plot_surface(X, Y, Z, cmap=cm.viridis)

fig2 = plt.figure(2, figsize=(8,6))
ax2 = fig2.add_subplot(1, 1, 1)
plot_contour = ax2.contourf(X, Y, Z, cmap=cm.viridis)

plt.show()


# MINIMA ######################################################################
i = 0
dif = [1,1]

x_new = [0,-0.5]
alpha = 0.1
epsilon = 0.0000000001
max_iterations = 300

while ((dif[0] > epsilon) or (dif[1] > epsilon)) and (i < max_iterations):
    x[0] = x_new[0]
    x[1] = x_new[1]
    g = grad(*x)

    x_new[0] += -alpha * g[0]
    x_new[1] += -alpha * g[1]

    dif[0] = abs(x[0] - x_new[0])
    dif[1] = abs(x[1] - x_new[1])
    
    i += 1

print "\n# Gradient directions algorithm #\n\nMinima: ", x_new, "\nIterations: ", i, "\n"
