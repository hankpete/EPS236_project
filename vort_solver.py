#!/usr/bin/env python3 
################################################################################
# Solve the vorticity equations for a 2D grid. 
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import random


def plot_scalar_field(data, title):
    """
    2D plot of a numpy array of scalars using imshow
    """
    f, ax = plt.subplots(1, figsize=(10,10))
    im = ax.imshow(data.T, origin='lower', cmap=plt.get_cmap('rainbow'))
    cb = f.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.show()


def plot_vector_field(vel, title):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10,10))
    u = vel[0].T
    v = vel[1].T
    skip = 4
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='mid', units='inches', scale=40, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5)
    qk = plt.quiverkey(Q, 0.5, 0.05, 1, '1', coordinates='figure')
    ax.set_title(title)
    plt.show()


def gradient_scalar(f):
    """
    Calculate gradient vector of scalar function
    """
    # use numpy's built in gradient function
    [f_x, f_y] = np.gradient(f, x, y)
    # periodic x boundaries
    f_x[0, :] = (f[1, :] - f[-1, :]) / (x[1] + LX - x[-1])
    f_x[-1, :] = (f[0, :] - f[-2, :]) / (x[0] + LX - x[-2])
    return np.array([f_x, f_y])


def gradient_vel(vel):
    """
    Calculate gradient vector for vel = [u, v]
    """
    [u_x, u_y] = gradient_scalar(vel[0])
    [v_x, v_y] = gradient_scalar(vel[1])
    return np.array([u_x, v_x]), np.array([u_y, v_y])


def solve_poisson(f):
    """
    Solve poisson equation 

        \Delta u = f

    given forcing term 'f' and the spacing of the grid
    """
    tol = 1e-8
    u = streamf
    error = 1
    error_mag = 0
    i = 0
    while error > tol:
        # Solutions should be the average of the gridpoints around it
        u_new = (dy**2 * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)) + \
                dx**2 * (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)) -  \
                dx**2 * dy**2 * f) / \
                (2 * dy**2 + 2 * dx**2)

        # Neumann BC for the periodic boundaries
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]

        # Dirichlet BC for the stress free boundaries
        u_new[:, 0] = 0 
        u_new[:, -1] = 0

        error = np.max(np.abs(u - u_new))
        u = u_new
        i += 1

        if error_mag != int(np.log10(error)):
            error_mag = int(np.log10(error))
            print("  err: {:1.8E}".format(error))
    # plot_scalar_field(f, "f in Poisson")
    # plot_scalar_field(u, "u in Poisson")
    return u


def take_step(vort, streamf):
    # calculate stream function \Delta Psi = -vort
    print("Calculating stream function.")
    streamf = solve_poisson(-vort)

    # update vorticity
    [vort_x, vort_y] = gradient_scalar(vort)
    [streamf_x, streamf_y] = gradient_scalar(streamf)
    [vort_xx, vort_xy] = gradient_scalar(vort_x)
    [vort_yx, vort_yy] = gradient_scalar(vort_y)

    vort += dt * (kin_visc * (vort_xx + vort_yy) - (vort_x * streamf_y) + (vort_y * streamf_x))

    vort[0, :] = vort[1, :]
    vort[-1, :] = vort[-2, :]
    vort[:, 0] = 0
    vort[:, -1] = 0

    return vort, streamf


### Run Solver
print("\nVorticity Equation Solver For 2D Incompressible Flow\n")

# NX by NY grid, dimensions LX by LY
NX = 128
NY = 128
LX = 8
LY = 6

x = np.linspace(0, LX, NX + 1)
x = x[:NX]
# y = (np.linspace(-1, 1, N)**3 + 1) / 2
y = np.linspace(-LY//2, LY//2, NY + 1)
y = y[:NY]
xx, yy = np.meshgrid(x, y)

dx = np.abs(np.roll(x, 1, axis=0) - x) 
dy = np.abs(np.roll(y, 1, axis=0) - y) 
dx[0] = LX - x[-1]
dy[0] = LY//2 - y[-1]

min_dx = np.min(np.abs(dx))
min_dy = np.min(np.abs(dy))

vel = np.zeros( (2, NX, NY) ) # u = vel[0], v = vel[1]
kick = 0.03
deltaU = 2
delta = 0.2
for i in range(NX):
    for j in range(NY):
        yj = y[j]
        r = random.random()
        vel[0, i, j] = deltaU / 2 * np.tanh(2 * yj / delta) + (r - 0.5) * kick * np.exp(-20 * yj**2)
vel_x, vel_y = gradient_vel(vel)
vort = vel_x[1] - vel_y[0]

press = np.ones( (NX, NY) )

density = np.ones( (NX, NY) )

streamf = np.zeros( (NX, NY) )

kin_visc = 1e-3

# Re = 1 / kin_visc

h = np.min([min_dx, min_dy])
u = np.mean(np.abs(vel[0]))
dtmax = np.min([h**2 / (4 * kin_visc), kin_visc / u])
print("dtmax: {}\n".format(dtmax))
dt = 0.01
print("dt: {}\n".format(dt))

i = 0
Nplot = 10
while True:
    print("Step: {} (t = {:2.4f})".format(i, dt * i))
    if i % Nplot == 0:
        plot_scalar_field(vort, "Vorticity, t = {:2.2f}".format(dt * i))

    vort, streamf = take_step(vort, streamf)
    i += 1
