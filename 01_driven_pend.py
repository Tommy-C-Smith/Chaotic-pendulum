#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:11:55 2024

@author: tommycursonsmith
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
q = 2
f =1.25
rel_driving_omega = 0.7

# RK4 step function
def rk4_step(grad, time, state, step_size):
    k1 = grad(time, state) * step_size
    k2 = grad(time + step_size / 2, state + k1 / 2) * step_size
    k3 = grad(time + step_size / 2, state + k2 / 2) * step_size
    k4 = grad(time + step_size, state + k3) * step_size
    return time + step_size, state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Gradient of the system
def grad(time, state):
    theta, omega = state[:2]  
    omega_dot = -omega / q - np.sin(theta) + f * np.cos(rel_driving_omega * time)
    return np.array([omega, omega_dot])

# Plotting function
def plot_simulation_and_phase_space(times, states, f, q, rel_driving_omega, lyapunov_exponent):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].plot(times, states[:, 0], label=r"$\theta$ (rad)",linewidth=2)
    axs[0].plot(times, states[:, 1], label=r"$\dot{\theta}$ (rad/s)",linewidth=2)
    axs[0].set_xlabel("Time (s)",fontsize=25)
    axs[0].set_ylabel("State",fontsize=25)
    axs[0].legend(fontsize=20)
    axs[0].set_title("Time Series Plot",fontsize=25)

    axs[1].plot(states[:, 0], states[:, 1],linewidth=3)
    axs[1].set_xlabel(r"$\theta$ (rad)",fontsize=25)
    axs[1].set_ylabel(r"$\dot{\theta}$ (rad/s)",fontsize=25)
    axs[1].set_title("Phase Space",fontsize=25)
    
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(2) 
            ax.tick_params(width=3)
            ax.tick_params(axis='both', which='major', labelsize=12, width=2)  
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)  

    params_text = f"f = {f}, q = {q}, rel_driving_omega = {rel_driving_omega}, Lyapunov = {lyapunov_exponent:.4f}"
    fig.text(0.5, -0.05, params_text, ha='center', fontsize=25)  # Adjust y position here
    
    plt.tight_layout(pad=3.0)
    
    plt.show()


# Main function
def main():
    # Initial conditions
    t_0 = 0
    t_end = 200
    step_size = 0.01
    initial_state = np.array([1, 0])  
    delta = 1e-6  
    n_steps = int((t_end - t_0) / step_size)
    rescale_interval = 50  

    # Prepare state arrays
    state_0 = initial_state
    state_1 = initial_state + np.array([delta, 0])

    log_distance_ratios = []

    # Simulation and Lyapunov exponent calculation
    times = np.array([t_0])
    states = np.array([initial_state])
    for step in range(n_steps):
        new_time, state_0 = rk4_step(grad, times[-1], state_0, step_size)
        _, state_1 = rk4_step(grad, times[-1], state_1, step_size)
        times = np.append(times, new_time)
        states = np.vstack((states, state_0))

        # Rescaling step for Lyapunov calculation
        if step % rescale_interval == 0 and step > 0:
            distance = np.linalg.norm(state_1 - state_0)
            log_distance_ratios.append(np.log(distance / delta))
            direction = (state_1 - state_0) / distance
            state_1 = state_0 + direction * delta

    lyapunov_exponent = np.mean(log_distance_ratios) / (step_size * rescale_interval)
    print(f"Maximal Lyapunov Exponent: {lyapunov_exponent}")

    plot_simulation_and_phase_space(times, states, f, q, rel_driving_omega, lyapunov_exponent)

if __name__ == "__main__":
    main()
