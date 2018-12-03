#!/usr/bin/env python3 
################################################################################
# Make plots from data from vorticity simulation
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
    im = ax.imshow(data.T, origin='lower', cmap=plt.get_cmap('plasma_r'), interpolation='none', extent=(x[0], x[-1], y[0], y[-1]), vmax=0)
    cb = f.colorbar(im, ax=ax, fraction=0.0355, pad=0.02, ticks=np.arange(-9, 1))

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
    data = -data
    levels = np.linspace(np.min(data), np.max(data), nLevels)
    ax.contour(xx, yy, data.T, levels, origin='lower', colors='b')

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
    cb = f.colorbar(im, ax=ax, fraction=0.0355, pad=0.02, ticks=np.arange(-9, 1))

    nLevels = 30
    data = -streamf
    levels = np.linspace(np.min(data), np.max(data), nLevels)
    ax.contour(xx, yy, data.T, levels, origin='lower', colors='w', linewidths=0.7)

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


def plot_vel_streamf(vel, streamf, title=""):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10, 8))

    nLevels = 15
    data = -streamf
    levels = np.linspace(np.min(data), np.max(data), nLevels)
    ax.contour(xx, yy, data.T, levels, origin='lower', colors='b')

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

    fname = "../images/vel_streamf{:03}.png".format(Nfig)
    print("  {} saved.".format(fname))
    plt.savefig(fname, dpi=100)
    plt.close()


def plot_vel_vort(vel, vort, title=""):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10, 8))

    im = ax.imshow(vort.T, origin='lower', cmap=plt.get_cmap('plasma_r'), interpolation='none', extent=(x[0], x[-1], y[0], y[-1]), vmax=0)
    cb = f.colorbar(im, ax=ax, fraction=0.0355, pad=0.02, ticks=np.arange(-9, 1))

    u = vel[0].T
    v = vel[1].T
    skip = 8
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='middle', units='inches', scale=40, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5, color='w')
    qk = plt.quiverkey(Q, 0.95, 0.95, 1, '1', coordinates='figure')

    ax.set_title(title, size=20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    plt.tight_layout()

    fname = "../images/vel_vort{:03}.png".format(Nfig)
    print("  {} saved.".format(fname))
    plt.savefig(fname, dpi=100)
    plt.close()


def plot_vel_vort_streamf(vel, vort, streamf, title=""):
    """
    2D plot of two numpy arrays of vector components using quiver
    """
    f, ax = plt.subplots(1, figsize=(10, 8))

    im = ax.imshow(vort.T, origin='lower', cmap=plt.get_cmap('plasma_r'), interpolation='none', extent=(x[0], x[-1], y[0], y[-1]), vmax=0)
    cb = f.colorbar(im, ax=ax, fraction=0.0355, pad=0.02, ticks=np.arange(-9, 1))

    nLevels = 30
    data = -streamf
    levels = np.linspace(np.min(data), np.max(data), nLevels)
    ax.contour(xx, yy, data.T, levels, origin='lower', colors='w', linewidths=0.7)

    u = vel[0].T
    v = vel[1].T
    skip = 8
    Q = ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], pivot='middle', units='inches', scale=40, scale_units='width', headwidth=3, headlength=4, headaxislength=3.5, color='w')
    qk = plt.quiverkey(Q, 0.95, 0.95, 1, '1', coordinates='figure')

    ax.set_title(title, size=20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    plt.tight_layout()

    fname = "../images/vel_vort_streamf{:03}.png".format(Nfig)
    print("  {} saved.".format(fname))
    plt.savefig(fname, dpi=100)
    plt.close()
    
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

Tplot = 0.5
Tprint = 0.25
for Nfig in range(101):
    data = np.load("../data/step{:03}.npz".format(Nfig))
    vort = data["vort"]
    streamf = data["streamf"]
    vel = data["vel"]

    t = Nfig * Tplot

    plot_scalar_field(vort, "Vorticity, t = {:2.1f}".format(t))

    plot_stream(streamf, "Stream Function, t = {:2.1f}".format(t))

    plot_vort_stream(vort, streamf, "Vorticity and Stream Function, t = {:2.1f}".format(t))

    plot_vector_field(vel, "Velocity Vectors, t = {:2.1f}".format(t))

    plot_vel_streamf(vel, streamf, "Velocity and Stream Function, t = {:2.1f}".format(t))

    plot_vel_vort(vel, vort, "Velocity and Vorticity, t = {:2.1f}".format(t))

    plot_vel_vort_streamf(vel, vort, streamf, "Velocity, Vorticity, and Stream Function, t = {:2.1f}".format(t))
