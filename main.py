##### Monte Carlo simulation of Path Integrals #####
from scipy import *
import matplotlib.pyplot as plt
import numpy.random as rndm
import sys

hbar = 1
m = 1
omega = 1

def harmonicWavefunction(x):
    alpha = m/2
    prefactor = (pi * hbar / (m * omega)) ** (-1/4)
    y = prefactor * exp(-alpha * (x ** 2))
    return y

def harmonicPotential(x):
    PE = m * sum((x ** 2) * (omega ** 2)) * 0.5
    return PE

def totalEnergy(path, eps):
    kineticEnergy = m * sum(((path[1:] - path[0:-1]) / eps) ** 2) * 0.5
    energy = kineticEnergy + harmonicPotential(path)
    return energy

def metropolis(newPath, path, eps):
    dEnergy = totalEnergy(newPath, eps) - totalEnergy(path, eps)

    if dEnergy == 0:
        print(totalEnergy(newPath, eps))
        print(totalEnergy(path,eps))
        print(path)
        print(newPath)
    if dEnergy < 0:
        return True
    elif rndm.rand() < exp(-eps * dEnergy):
        return True
    else: 
        return False

#rndm.seed(4401)        ### For reproducibility

N = 50                  ### Number of points in time
path = zeros(N)         ### Initialise path

MCsteps = 1000000       ### Number of Monte Carlo iterations
delta = 0.2             ### Change in path
eps = 0.1               ### Discrete time intervals

xmin = -3.2
xmax = 3.2 
xsteps = int((xmax - xmin) / delta)

threshold = int(0.2 * MCsteps)

pltInterval = int(0.01 * MCsteps)
fig = plt.figure(1)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

pathCount = 0
pathInterval = 25
time = arange(0, N * eps, eps)
xs = linspace(xmin, xmax, xsteps + 1)
xCount = zeros(len(xs))
accept = 0

for stepIdx in range(MCsteps):
    newPath = path.copy()
    timeIdx = rndm.randint(0, N)

    if path[timeIdx] == xmax:
        pathChange = -1
    elif path[timeIdx] == xmin:
        pathChange = +1
    else:
        pathChange = rndm.choice([-1, 1])

    newPath[timeIdx] += (pathChange * delta)

    if metropolis(newPath, path, eps):
        path = around(newPath, decimals = 1)
        accept += 1
        
    if stepIdx >= threshold - 1 and (stepIdx + 1) % pathInterval == 0 :
        for idx in range(len(path)):
            xIdx = int(rint((path[idx] - xmin) / delta))
            xCount[xIdx] += 1
        pathCount += 1

    if stepIdx % pltInterval == 1:
        ax1.clear()
        ax2.clear()
        ax1.plot(path, time)
        ax1.set_xlim([xmin - delta, xmax + delta])
        if pathCount > 0:
            ax2.scatter(xs, xCount / (pathCount * N * delta), marker = 'x')
        plt.draw()
        plt.pause(1e-17)

    percentDone = (stepIdx + 1) / MCsteps * 100
    done = int(percentDone / 2)
    
    sys.stdout.write("\r  [%s%s][%3.2f%%]" % ('=' * done, ' ' * (50 - done), percentDone))
    sys.stdout.flush()

print('')
ax1.clear()
ax2.clear()
ax1.plot(path, time)
ax1.set_xlim([xmin - delta, xmax + delta])
ax2.scatter(xs, xCount / (pathCount * N * delta), marker = 'x')
ax2.plot(xs, abs(harmonicWavefunction(xs)) ** 2)
plt.draw()
plt.show()