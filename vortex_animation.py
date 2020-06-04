import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# GRID SETTINGS

N = 120
x_start, x_end = -2.0, 2.0
y_start, y_end = -2.0, 2.0

x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)

X, Y = np.meshgrid(x, y)

# VORTEX FORCING (PRESSURE)

variance = 0.3   # radius
center = [0, 0]  # location
alpha = 1000.    # amplitude of

pos = np.dstack((X, Y))
p = 101500. - alpha*st.multivariate_normal.pdf(pos, mean=center, cov=np.eye(2) * variance)

# GAIN THE G. VELOCITY

f = 10**-4
rho = 1.

dpdx = np.gradient(p, 20000, axis=1)  # 1 is row in python
dpdy = np.gradient(p, 20000, axis=0)  # 0 is column in python


ubg_max = np.linspace(0, 10, 11)
levels = np.linspace(0, 13, 14)

for i in range(ubg_max.shape[0]):
    print("Current step is {:03d}; Current flow is {:5.2f}".format(i, ubg_max[i]))

    ubg1d = np.linspace(0.0, ubg_max[i], num=N)
    ubg = np.meshgrid(ubg1d)

    u = -1./(f*rho)*dpdy + ubg
    v = +1./(f*rho)*dpdx

    size = 12
    fig = plt.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size-2))
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.tick_params(axis="both", direction='in')
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)
    plt.grid(color='black', linestyle='--', linewidth=0.5)

    plt.title("$U_{{bg}}$ = {:5.2f} [m s-1]".format(ubg_max[i]))

    plt.contourf(X, Y, np.sqrt(u**2+v**2), levels=levels, extend='both', cmap='Blues')  # 20, vmin=0,vmax=13, extend
    cbar = plt.colorbar()
    plt.streamplot(X, Y, u, v, density=1, linewidth=1, arrowsize=1, arrowstyle='->', color='k')
    plt.scatter(center[0], center[1], color='#CD2305', s=20, marker='x')
    plt.savefig("out/CuetteFlow_%03d.png" % (i), dpi=150)
    plt.close()
