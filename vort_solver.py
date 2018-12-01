#!/usr/bin/env python3 
################################################################################
# Solve the vorticity equations for a 2D grid. 
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import random
import os


def plot_scalar_field(data, title=""):
    """
    2D plot of a numpy array of scalars using imshow
    """
    f, ax = plt.subplots(1, figsize=(10, 8))
    im = ax.imshow(data.T, origin='lower', cmap=plt.get_cmap('rainbow'), interpolation='none', extent=(x[0], x[-1], y[0], y[-1]))#, vmin=-8, vmax=8)
    cb = f.colorbar(im, ax=ax, fraction=0.0355, pad=0.02)

    ax.set_title(title, size=20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    cb.ax.tick_params(labelsize=12)

    plt.tight_layout()

    fname = "../images/vort{:03}.png".format(Nfig)
    print("  {} saved.".format(fname))
    plt.savefig(fname, dpi=100)
    plt.close()


def plot_stream(data, title=""):
    """
    2D plot of stream function using contours
    """
    f, ax = plt.subplots(1, figsize=(10, 8))
    nLevels = 25
    levels = np.linspace(np.min(data), np.max(data), nLevels)
    ax.contour(xx, yy, data.T, levels, origin='lower', colors='k')

    ax.set_title(title, size=20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    plt.tight_layout()

    fname = "../images/streamf{:03}.png".format(Nfig)
    print("  {} saved.".format(fname))
    plt.savefig(fname, dpi=100)
    plt.close()


def plot_vector_field(vel, title=""):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10, 8))
    u = vel[0].T
    v = vel[1].T
    skip = 8
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='middle', units='inches', scale=40, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5)
    qk = plt.quiverkey(Q, 0.95, 0.95, 1, '1', coordinates='figure')

    ax.set_title(title, size=20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    plt.tight_layout()

    fname = "../images/vel{:03}.png".format(Nfig)
    print("  {} saved.".format(fname))
    plt.savefig(fname, dpi=100)
    plt.close()



def gradient_scalar(f):
    """
    Calculate gradient vector of scalar function
    """
    # use numpy's built in gradient function
    # [f_x, f_y] = np.gradient(f, x, y)

    f_x = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    f_y = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

    # periodic x boundaries
    # f_x[0, :] = (f_x[1, :] - f_x[-2, :]) / (2 * dx)
    # f_x[-1, :] = f_x[0, :]
    # wall y boundaries
    f_y[:, 0] = f_y[:, 1]
    f_y[:, -1] = f_y[:, -2]
    return np.array([f_x, f_y])


def laplace_scalar(f):
    """
    Calculate laplacian of scalar function
    """
    f_xx = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2
    f_yy = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2
    # periodic x boundaries
    f_xx[-1, :] = f_xx[0, :]
    # wall y boundaries
    f_yy[:, 0] = f_yy[:, 1]
    f_yy[:, -1] = f_yy[:, -2]
    return f_xx + f_yy


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
    u = streamf
    error = 1
    error_mag = 0
    i = 0
    while error > poisson_tol:
        # Solutions should be the average of the gridpoints around it
        u_new = (dy**2 * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)) + \
                dx**2 * (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)) -  \
                dx**2 * dy**2 * f) / \
                (2 * dy**2 + 2 * dx**2)

        # Neumann BC for the periodic boundaries
        u_new[0, :] = u_new[1, :]
        # u_new[-1, :] = u_new[-2, :]
        u_new[-1, :] = u_new[0, :]

        # Dirichlet BC for the stress free boundaries
        u_new[:, 0] = 0 
        u_new[:, -1] = 0

        error = np.max(np.abs(u - u_new))
        u = u_new
        i += 1

        # if error_mag != int(np.log10(error)):
        #     error_mag = int(np.log10(error))
        #     print("  err: {:1.8E}".format(error))
    # plot_scalar_field(f, "f in Poisson")
    # plot_scalar_field(u, "u in Poisson")
    return u


def take_step(vort, streamf):
    # calculate stream function \Delta Psi = -vort
    streamf = solve_poisson(-vort)

    # update vorticity
    [vort_x, vort_y] = gradient_scalar(vort)
    [streamf_x, streamf_y] = gradient_scalar(streamf)
    [vort_xx, vort_yx] = gradient_scalar(vort_x)
    [vort_yx, vort_yy] = gradient_scalar(vort_y)

    vort += dt * (kin_visc * laplace_scalar(vort) - (vort_x * streamf_y) + (vort_y * streamf_x))
    # vort += dt * (kin_visc * (vort_xx + vort_yy) - (vort_x * streamf_y) + (vort_y * streamf_x))

    # vort[-1, :] = vort[0, :]
    vort[:, 0] = 0
    vort[:, -1] = 0

    return vort, streamf


def init_cond(kick, deltaU, delta):
    """
    Set the initial vorticity distribution
    """
    vel = np.zeros( (2, NX, NY) ) # u = vel[0], v = vel[1]
    for i in range(NX-1):
        for j in range(NY):
            yj = y[j]
            r = random.random()
            vel[0, i, j] = deltaU / 2 * np.tanh(2 * yj / delta) + (r - 0.5) * kick * np.exp(-20 * yj**2)
    vel[0, -1, :] = vel[0, 0, :]
    
    vel_x, vel_y = gradient_vel(vel)
    vort = vel_x[1] - vel_y[0]

    return vort
    
### Run Solver
print("\nVorticity Equation Solver For 2D Incompressible Flow\n")

random.seed(12345678987654321)

# NX by NY grid, dimensions LX by LY
NX = 128
NY = 2 * 128
LX = 8
LY = 6

x = np.linspace(0, LX, NX)
y = np.linspace(-LY//2, LY//2, NY)
xx, yy = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initial conditions and parameters
kin_visc = 1e-3

kick = 0.03; deltaU = 2; delta = 0.2
vort = init_cond(kick, deltaU, delta)

# streamf = np.zeros( (NX, NY) )
streamf = np.load("initial_streamf.npz")["streamf"]

h = np.min([dx, dy])
u = deltaU / 2
dtmax = np.min([h**2 / (4 * kin_visc), kin_visc / u])
print("dtmax: {}\n".format(dtmax))
dt = 0.5 * dtmax 

# Begin stepping
i = 0
Tplot = 0.5
Tprint = 0.25
Nplot = int(Tplot / dt)
Nprint = int(Tprint / dt)
Nfig = 0
poisson_tol = 1e-6
while True:
    if i % Nprint == 0:
        print("Step: {} (t = {:2.4f})".format(i, dt * i))
    if i % Nplot == 0:
        plot_scalar_field(vort, "Vorticity, t = {:2.1f}".format(dt * i))

        plot_stream(streamf, "Stream Function, t = {:2.1f}".format(dt * i ))

        streamf_x, streamf_y = gradient_scalar(streamf)
        vel = np.array([streamf_y, -streamf_x])
        plot_vector_field(vel, "Velocity Vectors, t = {:2.1f}".format(dt * i ))

        Nfig += 1

    vort, streamf = take_step(vort, streamf)
    # if i == 0:
    #     np.savez("initial_streamf.npz", streamf=streamf)
    i += 1
