#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:35:55 2024

@author: tommycursonsmith
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
q = 6
rel_driving_omega = 1.23
step_size = 0.01
n_steps = 40_000  # Large enough to ensure reaching steady state
transient_steps = 10_000  # Number of steps to skip for transient behavior

# Simulation functions as provided, with slight modifications if necessary
def rk4_step(grad, time, state, step_size, f):
    """
    Perform a single step of RK4, with dependency on dynamic parameter f.
    """
    k1 = grad(time, state, f) * step_size
    k2 = grad(time + step_size / 2, state + k1 / 2, f) * step_size
    k3 = grad(time + step_size / 2, state + k2 / 2, f) * step_size
    k4 = grad(time + step_size, state + k3, f) * step_size
    return time + step_size, state + (k1 + 2*k2 + 2*k3 + k4) / 6

def grad(time, state, f):
    """Gradient function, now f is a parameter."""
    theta, omega = state
    theta_dot = omega
    omega_dot = -omega / q - np.sin(theta) + f * np.cos(rel_driving_omega * time)
    return np.array([theta_dot, omega_dot])

def simulate_system(f, n_steps, transient_steps):
    """Simulate the system for a given f, ignoring transients."""
    state = np.array([1, 0])  # Initial state: theta, omega
    time = 0
    states = []
    for i in range(n_steps):
        time, state = rk4_step(grad, time, state, step_size, f)
        if i >= transient_steps:  # Collect data after transient phase
            states.append(state)
    return np.array(states)

def find_local_extremes(states):
    """Find local maxima and minima of omega from states."""
    omegas = states[:, 1]
    maxima = (np.diff(np.sign(np.diff(omegas))) < 0).nonzero()[0] + 1
    minima = (np.diff(np.sign(np.diff(omegas))) > 0).nonzero()[0] + 1
    return omegas[maxima], omegas[minima]

def plot_bifurcation(fs, maxima, minima):
    """Plot the bifurcation diagram."""
    plt.figure(figsize=(10, 6))
    for f, max_vals, min_vals in zip(fs, maxima, minima):
        plt.plot([f] * len(max_vals), max_vals, 'ro', markersize=2, label='Maxima' if f == fs[0] else "")
        plt.plot([f] * len(min_vals), min_vals, 'bo', markersize=2, label='Minima' if f == fs[0] else "")
    plt.xlabel('Amplitude parameter (f)',fontsize=17)
    plt.ylabel('Angular Velocity of local extremes (rad/s)',fontsize=17)
    plt.title('Amplitude Bifurcation Plot',fontsize=22)
    plt.legend(fontsize=15)
    plt.show()

# Main experiment setup
fs = np.linspace(0.5, 2.5, 500)  # Range of f values
maxima_list = []
minima_list = []
for f in fs:
    states = simulate_system(f, n_steps, transient_steps)
    maxima, minima = find_local_extremes(states)
    maxima_list.append(maxima)
    minima_list.append(minima)

plot_bifurcation(fs, maxima_list, minima_list)
