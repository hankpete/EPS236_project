#!/usr/bin/env python3 
################################################################################
#  Solve the N-S equations for a 2D grid.
################################################################################

import numpy as np
import matplotlib.pyplot as plt


def solve_poisson(f, dx, dy):
    """
    Solve poisson equation 

        -\Delta u = f

    given forcing term 'f' and the spacing of the grid
    """

    print("Solving Poisson...")

    tol = 1e-5
    u = np.ones(f.shape)
    error = 1
    i = 0
    while error > tol:
        # Solutions should be the average of the gridpoints around it
        u_new = 1/4 * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) + f * dx**2)

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


def take_step(vel, press, density, kin_visc, dt, dx, dy):
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
    vel_x = (np.roll(vel, 1, axis=1) - vel) / dx
    vel_y = (np.roll(vel, 1, axis=2) - vel) / dy
    vel_xx = (np.roll(vel, 1, axis=1) - 2 * vel + np.roll(vel, -1, axis=1)) / dx**2
    vel_yy = (np.roll(vel, 1, axis=2) - 2 * vel + np.roll(vel, -1, axis=2)) / dy**2

    press_x = (np.roll(press, 1, axis=0) - press) / dx
    press_y = (np.roll(press, 1, axis=1) - press) / dy
    grad_press = np.array([press_x, press_y])

    # take step
    vel_star = vel + dt * ( -(vel[0] * vel_x + vel[1] * vel_y) - beta * grad_press / density + kin_visc * (vel_xx + vel_yy) )

    # stress free boundary conditions at the top and bottom (periodic on left and right)
    vel_star[:, :, 0] = vel_star[:, :, 1] 
    vel_star[:, :, -1] = vel_star[:, :, -2]

    ### (2) solve poisson for phi
    # get gradients
    vel_star_x = (np.roll(vel_star, 1, axis=1) - vel_star) / dx
    vel_star_y = (np.roll(vel_star, 1, axis=2) - vel_star) / dy
    # solve \Delta phi = density/dt * div(vel)
    phi = solve_poisson(density / dt * (vel_star_x[0] + vel_star_y[1]), dx, dy)

    ### (3) correct vel_star with phi
    # get gradients
    phi_x = (np.roll(phi, 1, axis=0) - phi) / dx
    phi_y = (np.roll(phi, 1, axis=1) - phi) / dy
    grad_phi = np.array([phi_x, phi_y])
    # incompressibility correction
    vel = vel_star - dt / density * grad_phi 

    # stress free boundary conditions at the top and bottom (periodic on left and right)
    vel[:, :, 0] = vel[:, :, 1] 
    vel[:, :, -1] = vel[:, :, -2]
    
    ### (4) correct press with phi
    press = phi + beta * press

    return vel, press


def plot_scalar_field(data):
    """
    2D plot of a numpy array of scalars using imshow
    """
    f, ax = plt.subplots(1, figsize=(10,10))
    im = ax.imshow(data.T, origin='lower')
    cb = f.colorbar(im, ax=ax)
    plt.show()


def plot_vector_field(data, xx, yy):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10,10))
    u = data[0].T
    v = data[1].T
    skip = 4
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='mid', units='inches', scale=30, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5)
    qk = plt.quiverkey(Q, 0.5, 0.05, 1, '1', coordinates='figure')
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    plt.show()


def main():
    """
    Run the solver
    """

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
    dxx, dyy = np.meshgrid(dx, dy)

    min_dx = np.min(np.abs(dx))
    min_dy = np.min(np.abs(dy))

    # # Each point had a dx and a dy associated with it
    # high = 0.1; low = 1.0
    # dx = low * np.ones((N, N))
    # dy = low * np.ones((N, N))
    # # Make dy finer at the interface
    # for x in range(N):
    #     dy[x, :] = 2/3*high + 1/3*low - 2/3 * (high-low) * 1/2 * (3 * np.linspace(-1, 1, N)**2 - 1)

    vel = np.zeros( (2, N, N) ) # u = vel[0], v = vel[1]
    vel[0, :, :N//2] = -1
    vel[0, :, N//2:] = 1
    vel[0, (N//2-2):(N//2+2), N//2:(N//2+2)] = -1

    press = np.ones( (N, N) )

    density = np.ones( (N, N) )

    kin_visc = 1
    x = x[:N]

    dtmax = np.min([min_dx, min_dy])**2 / (2 * kin_visc + np.mean(np.abs(vel)) * np.min([min_dx, min_dy]))
    dt = 0.5 * dtmax

    i = 0
    while True:
        # if i % 100 == 0:
        if True:
            print("Step: {}".format(i))
            plot_vector_field(vel, xx, yy)

            vel_x = (np.roll(vel, 1, axis=1) - vel) / dxx
            vel_y = (np.roll(vel, 1, axis=2) - vel) / dyy
            vort = vel_x[1] - vel_y[0]
            plot_scalar_field(vort)

        vel, press = take_step(vel, press, density, kin_visc, dt, dxx, dyy)
        i += 1


main()
