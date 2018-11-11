#!/usr/bin/env python3 
################################################################################
# Solve the vorticity equations for a 2D grid. 
################################################################################

import numpy as np
import matplotlib.pyplot as plt


def solve_laplace(f, dx, dy):
    tol = 1e-5
    u = np.ones(f.shape)
    error = 1
    i = 0
    while error > tol:
        u_new = 1/4 * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) + f * dx**2)
        # u_new[0, :] = u_new[1, :]; u_new[-1, :] = u_new[-2, :]
        # u_new[:, 0] = u_new[:, 1]; u_new[:, -1] = u_new[:, -2]
        u_new[0, :] = 0; u_new[-1, :] = 0
        u_new[:, 0] = 0; u_new[:, -1] = 0
        error = np.max(np.abs(u - u_new))
        u = u_new
        i += 1

        # if i % 1e3 == 0:
            # print(error)
            # plot_image(u)
    return u


def take_step(vort, dt, Re, dx, dy):
    # calculate stream function -Delta Psi = omega
    streamf = solve_laplace(vort, dx, dy)

    # update vorticity
    [vort_x, vort_y] = np.gradient(vort, dx, dy)
    [streamf_x, streamf_y] = np.gradient(streamf, dx, dy)
    [vort_xx, vort_xy] = np.gradient(vort_x, dx, dy)
    [vort_yx, vort_yy] = np.gradient(vort_y, dx, dy)

    vort += dt * (1/Re * (vort_xx + vort_yy) - (vort_x * streamf_y) + (vort_y * streamf_x))
    vort[0, :] = 0; vort[-1, :] = 0
    vort[:, 0] = 0; vort[:, -1] = 0


def plot_image(data):
    f, ax = plt.subplots(1, figsize=(10,10))
    im = ax.imshow(data.T, origin='lower')
    cb = f.colorbar(im, ax=ax)
    plt.show()


def main():
    dx = 1; dy = 1;
    Re = 2000;
    dt = 0.5
    N = 100

    vort = np.zeros( (N, N) )
    vort[5:N-5, (N//2 - 1):(N//2 + 1)] = 0.1
    vort[(N//2 - 1):(N//2 + 1), (N//2-2)] = 0.1

    i = 0
    while True:
        if i % 10 == 0:
            plot_image(vort)

        take_step(vort, dt, Re, dx, dy)
        i += 1
        print(i)

    

main()




    
