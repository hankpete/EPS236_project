#!/usr/bin/env python3 
################################################################################
# This file is just for testing basic methods in 1D as a debugging tool. 
################################################################################

import numpy as np
import matplotlib.pyplot as plt


def solve_heat(f, dx, dy):
    tol = 1e-5
    u = 1000*np.ones(f.shape)
    stability = 0.5
    dt = stability * np.min([dx, dy])**2
    error = 1
    i = 0
    while error > tol:
        # u_xx = (np.roll(u, 1, axis=0) - 2 * u + np.roll(u, -1, axis=0)) / dx**2
        u_xx = np.gradient(np.gradient(u, dx), dx)
        u_new = u + dt * (u_xx + f)
        u_new[0] = u[0] + dt * (2 * u[1] - 2 * u[0])
        u_new[-1] = 0 #u[-1] + dt * (2 * u[-2] - 2 * u[-1])
        # u_new[0] = u_new[1]; u_new[-1] = u_new[-2] 
        error = np.max(np.abs((u_new - u)/dt))
        u = u_new
        i += 1

        if i % 1e3 == 0:
            print(error)
            # plot_image([u_xx, f, u_xx+f])
    plot_image([u, f])
    return u


def solve_laplace(f, dx, dy):
    tol = 1e-5
    u = np.copy(f)
    error = 1
    i = 0
    while error > tol:
        u_new = (f * dx**2) / 2 - (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)) / 2
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]
        error = np.max(np.abs(u - u_new))
        u = u_new
        i += 1

        if i % 1e3 == 0:
            print(error)
            # plot_image([u, f])
    return u


def take_step(vort, dt, Re, dx, dy):
    # calculate stream function -Delta Psi = omega
    # streamf = solve_laplace(vort, dx, dy)
    streamf = solve_heat(vort, dx, dy)

    # update vorticity
    # [vort_x, vort_y] = np.gradient(vort, dx, dy)
    # [streamf_x, streamf_y] = np.gradient(streamf, dx, dy)
    # [vort_xx, vort_xy] = np.gradient(vort_x, dx, dy)
    # [vort_yx, vort_yy] = np.gradient(vort_y, dx, dy)

    # dvort += dt * (1/Re * (vort_xx + vort_yy) - (vort_x * streamf_y) + (vort_y * streamf_x))



def plot_image(data):
    f, ax = plt.subplots(1, figsize=(10,10))
    for d in data:
        ax.plot(d)
    plt.show()


def main():
    dx = 1; dy = 1;
    Re = 100;
    dt = 0.1
    N = 100

    mu = 0
    sig = 5
    # vort = np.exp(-(np.arange(N) - N//2 - mu)**2 / (2*sig**2)) / np.sqrt(2 * np.pi * sig**2)
    vort = 1 - np.cos(5 * np.arange(N) * 2 * np.pi / N)
    plot_image([vort])
    while True:
        take_step(vort, dt, Re, dx, dy)
    

main()




    
