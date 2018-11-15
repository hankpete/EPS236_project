#!/usr/bin/env python3 
################################################################################
#  Solve the N-S equations for a 2D grid.
################################################################################

import numpy as np
import matplotlib.pyplot as plt


def plot_scalar_field(data):
    """
    2D plot of a numpy array of scalars using imshow
    """
    f, ax = plt.subplots(1, figsize=(10,10))
    im = ax.imshow(data.T, origin='lower')
    cb = f.colorbar(im, ax=ax)
    plt.show()


def plot_vector_field(vel):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10,10))
    u = vel[0].T
    v = vel[1].T
    skip = 4
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='mid', units='inches', scale=30, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5)
    qk = plt.quiverkey(Q, 0.5, 0.05, 1, '1', coordinates='figure')
    plt.show()


def gradient_scalar(f):
    """
    Calculate gradient vector of scalar function
    """
    # use numpy's built in gradient function
    [f_x, f_y] = np.gradient(f)
    # make sure we have periodic x boundaries
    # f_x[0, :] = (f[1, :] - f[0, :]) / (x[1] - x[0])
    f_x[-1, :] = (f[0, :] - f[-1, :]) / (x[0] + L - x[-1])
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

    print("Solving Poisson...")

    tol = 1e-5
    u = np.ones(f.shape)
    error = 1
    i = 0
    while error > tol:
        # Solutions should be the average of the gridpoints around it
        u_new = (dy**2 * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)) + \
                dx**2 * (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)) -  \
                dx**2 * dy**2 * f) / \
                (2 * dy**2 + 2 * dx**2)

        # Neumann BC for the periodic boundaries
        u_new[0, :] = u_new[1, :]; 
        # u_new[-1, :] = u_new[-2, :]

        # Dirichlet BC for the stress free boundaries
        u_new[:, 0] = 0; 
        u_new[:, -1] = 0

        error = np.max(np.abs(u - u_new))
        u = u_new
        i += 1

        if i % 1e3 == 0:
            print("  {:1.8E}".format(error))
    # plot_scalar_field(f)
    # plot_scalar_field(u)
    return u


def take_step(vel, press, density):
    """
    Take one time step in solving N-S. The procedure:
        (1) a preliminary explicit step
        (2) solve for the new pressure assuming incompressibility correcting factor
        (3) correct the velocity from (1) using result from (2)
        (4) correct the pressure using result from (2)
    """
    ### (1) Calculate intermediate velocity, vel_star
    # fractional step coeff
    beta = 1

    # get all the gradients (remember, vel contains both u and v)
    # vel_x = (np.roll(vel, 1, axis=1) - vel) / dx
    # vel_y = (np.roll(vel, 1, axis=2) - vel) / dy
    vel_x, vel_y = gradient_vel(vel)
    # vel_xx = (np.roll(vel, 1, axis=1) - 2 * vel + np.roll(vel, -1, axis=1)) / dx**2
    # vel_yy = (np.roll(vel, 1, axis=2) - 2 * vel + np.roll(vel, -1, axis=2)) / dy**2
    vel_xx, vel_yx = gradient_vel(vel_x)
    vel_yx, vel_yy = gradient_vel(vel_y)

    # press_x = (np.roll(press, 1, axis=0) - press) / dx
    # press_y = (np.roll(press, 1, axis=1) - press) / dy
    # grad_press = np.array([press_x, press_y])
    grad_press = gradient_scalar(press)

    # take step
    vel_star = vel + dt * ( -(vel[0] * vel_x + vel[1] * vel_y) - beta * grad_press / density + kin_visc * (vel_xx + vel_yy) )

    # stress free boundary conditions at the top and bottom (periodic on left and right)
    vel_star[:, :, 0] = vel_star[:, :, 1] 
    vel_star[:, :, -1] = vel_star[:, :, -2]

    ### (2) solve poisson for phi
    # get gradients
    # vel_star_x = (np.roll(vel_star, 1, axis=1) - vel_star) / dx
    # vel_star_y = (np.roll(vel_star, 1, axis=2) - vel_star) / dy
    vel_star_x, vel_star_y = gradient_vel(vel_star)
    # solve \Delta phi = density/dt * div(vel)
    phi = solve_poisson(density / dt * (vel_star_x[0] + vel_star_y[1]))

    ### (3) correct vel_star with phi
    # get gradients
    # phi_x = (np.roll(phi, 1, axis=0) - phi) / dx
    # phi_y = (np.roll(phi, 1, axis=1) - phi) / dy
    # grad_phi = np.array([phi_x, phi_y])
    grad_phi = gradient_scalar(phi)
    # incompressibility correction
    vel = vel_star - dt / density * grad_phi 

    # stress free boundary conditions at the top and bottom (periodic on left and right)
    vel[:, :, 0] = vel[:, :, 1] 
    vel[:, :, -1] = vel[:, :, -2]
    
    ### (4) correct press with phi
    press = phi + beta * press

    return vel, press


### Run Solver
print("Navier-Stokes Solver For 2D Incompressible Flow\n")

# NxN grid
N = 100
L = 1

x = np.linspace(0, L, N + 1)
x = x[:N]
# y = (np.linspace(-1, 1, N)**3 + 1) / 2
y = np.linspace(0, L, N + 1)
y = y[:N]
xx, yy = np.meshgrid(x, y)

dx = np.abs(np.roll(x, 1, axis=0) - x) 
dy = np.abs(np.roll(y, 1, axis=0) - y) 
dx[0] = x[0] + (L - x[-1])
dy[0] = y[0] + (L - y[-1])

min_dx = np.min(np.abs(dx))
min_dy = np.min(np.abs(dy))

vel = np.zeros( (2, N, N) ) # u = vel[0], v = vel[1]
vel[0, :, :N//2] = 1
vel[0, :, N//2:] = -1
vel[0, (N//2-2):(N//2+2), N//2:(N//2+2)] = 1

press = np.ones( (N, N) )

density = np.ones( (N, N) )

kin_visc = 1
x = x[:N]

dtmax = np.min([min_dx, min_dy])**2 / (2 * kin_visc + np.mean(np.abs(vel)) * np.min([min_dx, min_dy]))
dt = dtmax

i = 0
while True:
    print("Step: {}".format(i))
    if i % 10 == 0:
        plot_vector_field(vel)

        # vel_x = (np.roll(vel, 1, axis=1) - vel) / dxx
        # vel_y = (np.roll(vel, 1, axis=2) - vel) / dyy
        vel_x, vel_y = gradient_vel(vel)
        vort = vel_x[1] - vel_y[0]
        plot_scalar_field(vort)

    vel_new, press_new = take_step(vel, press, density)
    print(np.max(np.abs((vel_new-vel)/dt)))
    print(np.max(np.abs((press_new-press)/dt)))
    vel = vel_new
    i += 1
