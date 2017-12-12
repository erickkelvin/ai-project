import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return ((4 -(2.1*(x**2))) + ((1/3.0)*(x**4)))*(x**2) + (x*y) - 4*(1 - y**2)*(y**2)

def grad(*x):
    return [(2*(x[0]**5) - 8.4*(x[0]**3) + 8*x[0] + x[1]), (x[0] + 16*(x[1]**3) - 8*x[1])]

def plot_initial():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
    plot_3d = ax1.plot_surface(X, Y, Z, cmap=cm.viridis)

    fig2 = plt.figure(2, figsize=(8,6))
    ax2 = fig2.add_subplot(1, 1, 1)
    plot_contour = ax2.contourf(X, Y, Z, cmap=cm.viridis)

    plt.show()

def minimize(initial, step=0.01, tolerance=0.00001, max_iterations=200):
    k = 0
    dif = [1, 1]

    x_new = np.array(initial)
    x = np.array(initial)
    
    res = [(np.array(x_new), f(x_new[0],x_new[1]))]

    while ((dif[0] > tolerance) or (dif[1] > tolerance)) and (k < max_iterations):
        x[0] = x_new[0]
        x[1] = x_new[1]
        g = grad(*x)

        x_new[0] += step * -g[0]
        x_new[1] += step * -g[1]

        dif[0] = abs(x[0] - x_new[0])
        dif[1] = abs(x[1] - x_new[1])
        k += 1
        res += [(np.array(x_new), f(x_new[0],x_new[1]))]

    return res

def print_results(res):
    print ("\n# GRADIENT DIRECTIONS # \n\n# Iterations:\n")
    for k in range(0,len(res)):
        if (k<8):
            print ("[ k =",k,"]\t x: ", res[k][0],  " f(x):", res[k][1])
        if (k>len(res)-6):
            if (k>=(len(res)-2)):
                print ("[ k =",k,"]\t x: ", res[k][0],  " f(x):", res[k][1])
            else:
                print(".")

    print ("\n\n# Final results: \n\
        \nMinimiser x:\t", res[(len(res)-1)][0], "\
        \nMinimum f: \t", res[(len(res)-1)][1],"\
        \nIterations:\t", len(res)-1, "\n")

def plot_results(res):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig3 = plt.figure(figsize=(8,6))
    ax3 = fig3.add_subplot(1, 1, 1)
    plot_contour2 = plt.contourf(X, Y, Z)

    ax3.annotate("",
                xy=res[0][0], xycoords='data',
                xytext=res[len(res)-1][0], textcoords='data',
                arrowprops=dict(arrowstyle="-",
                                color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None,
                                patchB=None,
                                connectionstyle="arc3, rad=0.2"))
                        
    ax3.annotate("x0", xy=res[0][0])
    ax3.annotate("x"+str(len(res)-1), xy=res[len(res)-1][0])

    plt.show()


res = minimize([0, -0.5]) # call minimization function on defined initial point

print_results(res) # print table with initial and final iterations

plot_initial() # plot initial function (surface and contour)

plot_results(res) # plot function with convergence history line
