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
    im = ax.imshow(data.T, origin='lower', cmap=plt.get_cmap('rainbow'), interpolation='none', extent=(x[0], x[-1], y[0], y[-1]), vmax=0)
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


def plot_vort_stream(vort, streamf, title=""):
    """
    2D plot of a both vort and streamf
    """
    f, ax = plt.subplots(1, figsize=(10, 8))

    im = ax.imshow(vort.T, origin='lower', cmap=plt.get_cmap('plasma_r'), interpolation='none', extent=(x[0], x[-1], y[0], y[-1]), vmax=0)
    cb = f.colorbar(im, ax=ax, fraction=0.0355, pad=0.02)

    nLevels = 15
    data = -streamf
    # levels = np.linspace(np.min(data), np.max(data), nLevels)
    # ax.contour(xx, yy, data.T, levels, origin='lower', colors='k')
    ax.contour(xx, yy, data.T, origin='lower', colors='k')

    ax.set_title(title, size=20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    cb.ax.tick_params(labelsize=12)

    plt.tight_layout()

    fname = "../images/vort_streamf{:03}.png".format(Nfig)
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
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='middle', units='inches', scale=30, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5)
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
    return u


def take_step(vort, streamf):
    # calculate stream function \Delta Psi = -vort
    streamf = solve_poisson(-vort)

    # update vorticity
    [vort_x, vort_y] = gradient_scalar(vort)
    [streamf_x, streamf_y] = gradient_scalar(streamf)

    vort += dt * (kin_visc * laplace_scalar(vort) - (vort_x * streamf_y) + (vort_y * streamf_x))

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
NX = 2 * 128
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
# streamf = np.load("initial_streamf_128x256.npz")["streamf"]
streamf = np.load("initial_streamf_256x256.npz")["streamf"]

h = np.min([dx, dy])
u_max = deltaU / 2 + 0.5 * kick
v_max = 0
dtmax = np.min([h**2 / (4 * kin_visc), 2 * kin_visc / (u_max + v_max)])
dt_mult = 0.5
dt = dt_mult * dtmax 

# Begin stepping
i = 0
t = 0
Tplot = 0.5
Tprint = 0.25
Nfig = 0
poisson_tol = 1e-5
while True:
    # Print Status
    if t % Tprint < dt and i != 1:
        print("Step: {} (t = {:2.4f}, dt = {:1.2E})".format(i, t, dt))

    # Plot Status
    if t % Tplot < dt and i != 1:
        plot_scalar_field(vort, "Vorticity, t = {:2.1f}".format(t))

        plot_stream(streamf, "Stream Function, t = {:2.1f}".format(t))

        plot_vort_stream(vort, streamf, "Vorticity and Stream Function, t = {:2.1f}".format(t))

        streamf_x, streamf_y = gradient_scalar(streamf)
        vel = np.array([streamf_y, -streamf_x])
        plot_vector_field(vel, "Velocity Vectors, t = {:2.1f}".format(t))

        # fname = "../data/step{:03}.npz".format(Nfig)
        # np.savez(fname, vort=vort, streamf=streamf, vel=vel)
        # print("{} saved.".format(fname))

        Nfig += 1

    # Step forward in time
    vort, streamf = take_step(vort, streamf)
    t += dt
    
    # Recalculate dtmax
    streamf_x, streamf_y = gradient_scalar(streamf)
    u_max = np.max(np.abs(streamf_x))
    v_max = np.max(np.abs(streamf_y))
    dtmax = np.min([h**2 / (4 * kin_visc), 2 * kin_visc / (u_max + v_max)])
    dt = dt_mult * dtmax

    # if i == 0:
    #     np.savez("initial_streamf_256x256.npz", streamf=streamf)

    i += 1
