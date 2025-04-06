#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:42:37 2024

@author: tommycursonsmith
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the RK4 step as before
def rk4_step(grad, time, state, step_size):
    k1 = grad(time, state) * step_size
    k2 = grad(time + step_size / 2, state + k1 / 2) * step_size
    k3 = grad(time + step_size / 2, state + k2 / 2) * step_size
    k4 = grad(time + step_size, state + k3) * step_size
    return time + step_size, state + (k1 + 2*k2 + 2*k3 + k4) / 6

# Define your system's gradient function
q = 2.0; f = 1.8; rel_driving_omega = 1
def grad(time, state):
    theta, omega = state
    theta_dot = omega
    omega_dot = -omega/q - np.sin(theta) + f*np.cos(rel_driving_omega*time)
    return np.array([theta_dot, omega_dot])

# Initialize parameters
t_0 = 0
t_end = 200
step_size = 0.01
initial_state = np.array([4, 1])  # Initial state [theta, omega]
delta = 1e-6  # Small difference in initial conditions
n_steps = int((t_end - t_0) / step_size)
rescale_interval = 50  # Rescale every 100 steps

# Initialize states
state_0 = initial_state
state_1 = initial_state + np.array([delta, 0])  # Small perturbation in theta

# Store the logarithm of the distance ratio
log_distance_ratios = []

# Simulation loop
for step in range(n_steps):
    # Evolve both systems
    _, state_0 = rk4_step(grad, t_0 + step * step_size, state_0, step_size)
    _, state_1 = rk4_step(grad, t_0 + step * step_size, state_1, step_size)

    # Rescaling step
    if step % rescale_interval == 0 and step > 0:
        # Calculate the distance between the states in phase space
        distance = np.linalg.norm(state_1 - state_0)
        log_distance_ratios.append(np.log(distance / delta))
        # Rescale state_1 towards state_0
        direction = (state_1 - state_0) / distance
        state_1 = state_0 + direction * delta

# Calculate the Lyapunov exponent
lyapunov_exponent = np.mean(log_distance_ratios) / (step_size * rescale_interval)
print(f"Maximal Lyapunov Exponent: {lyapunov_exponent}")

# Optional: Plot the log of distance ratios over time
plt.plot(log_distance_ratios)
plt.xlabel('Rescale Step')
plt.ylabel('Log Distance Ratio')
plt.title('Log of Distance Ratios Over Time')
plt.show()
