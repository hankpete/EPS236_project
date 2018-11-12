#!/usr/bin/env python3 
################################################################################
#  Solve the N-S equations for a 2D grid.
################################################################################

import numpy as np
import matplotlib.pyplot as plt


def solve_poisson(f, dx, dy):
    tol = 1e-5
    u = np.ones(f.shape)
    error = 1
    i = 0
    while error > tol:
        u_new = 1/4 * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) + f * dx**2)

        u_new[0, :] = u_new[1, :]; 
        u_new[-1, :] = u_new[-2, :]
        # u_new[:, 0] = u_new[:, 1]; 
        # u_new[:, -1] = u_new[:, -2]

        # u_new[0, :] = 0; 
        # u_new[-1, :] = 0
        u_new[:, 0] = 0; 
        u_new[:, -1] = 0

        error = np.max(np.abs(u - u_new))
        u = u_new
        i += 1

        if i % 1e3 == 0:
            print(error)
            # plot_scalar_field(u)
    return u


def take_step(vel, press, density, kin_visc, dt, dx, dy):
    # Calculate intermediate velocity, vel_star
    beta = 1

    vel_x = (np.roll(vel, 1, axis=1) - vel) / dx
    vel_y = (np.roll(vel, 1, axis=2) - vel) / dy
    vel_xx = (np.roll(vel, 1, axis=1) - 2 * vel + np.roll(vel, -1, axis=1)) / dx**2
    vel_yy = (np.roll(vel, 1, axis=2) - 2 * vel + np.roll(vel, -1, axis=2)) / dy**2

    press_x = (np.roll(press, 1, axis=0) - press) / dx
    press_y = (np.roll(press, 1, axis=1) - press) / dy
    grad_press = np.array([press_x, press_y])

    vel_star = vel + dt * ( -(vel[0] * vel_x + vel[1] * vel_y) - beta * grad_press / density + kin_visc * (vel_xx + vel_yy) )

    vel_star[:, 0] = vel_star[:, 1]; 
    vel_star[:, -1] = vel_star[:, -2]

    # solve poisson for phi
    vel_star_x = (np.roll(vel_star, 1, axis=1) - vel_star) / dx
    vel_star_y = (np.roll(vel_star, 1, axis=2) - vel_star) / dy
    phi = solve_poisson(density / dt * (vel_star_x[0] + vel_star_y[1]), dx, dy)

    # correct vel_star with phi
    phi_x = (np.roll(phi, 1, axis=0) - phi) / dx
    phi_y = (np.roll(phi, 1, axis=1) - phi) / dy
    grad_phi = np.array([phi_x, phi_y])
    vel = vel_star - dt / density * grad_phi 

    vel[:, 0] = vel[:, 1]; 
    vel[:, -1] = vel[:, -2]
    

    # correct press with phi
    press = phi + beta * press

    return vel, press


def plot_scalar_field(data):
    f, ax = plt.subplots(1, figsize=(10,10))
    im = ax.imshow(data.T, origin='lower')
    cb = f.colorbar(im, ax=ax)
    plt.show()


def plot_vector_field(data):
    f, ax = plt.subplots(1, figsize=(10,10))
    u = data[0].T
    v = data[1].T
    skip = 2
    Q = ax.quiver(u[::skip, ::skip], v[::skip, ::skip], pivot='mid', units='inches', scale=25, scale_units='width', headwidth=6, headlength=4, headaxislength=3)
    qk = plt.quiverkey(Q, 0.5, 0.05, 2, '2', coordinates='figure')
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    plt.show()


def main():
    dx = 1; dy = 1;
    N = 100

    vel = np.zeros( (2, N, N) ) # u = vel[0], v = vel[1]
    vel[0, :, :N//2] = -1
    vel[0, :, N//2:] = 1
    vel[0, (N//2-2):(N//2+2), N//2:(N//2+2)] = -1

    press = np.ones( (N, N) )

    density = np.ones( (N, N) )

    kin_visc = 1

    dtmax = np.min([dx, dy])**2 / (2 * kin_visc + np.mean(vel) * np.min([dx, dy]))
    print(dtmax)
    dt = 0.1 * dtmax

    i = 0
    while True:
        # if i % 100 == 0:
        if True:
            plot_vector_field(vel)

            vel_x = (np.roll(vel, 1, axis=1) - vel) / dx
            vel_y = (np.roll(vel, 1, axis=2) - vel) / dy
            vort = vel_x[1] - vel_y[0]
            plot_scalar_field(vort)

        vel, press = take_step(vel, press, density, kin_visc, dt, dx, dy)
        i += 1


main()




    
