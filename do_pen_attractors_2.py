#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:21:56 2024

@author: tommycursonsmith
"""

import numpy as np
import matplotlib.pyplot as plt

def rk4_step(grad, time, state, step_size):
    """
    Performs a single Runge-Kutta 4 step.
    """
    k1 = grad(time, state) * step_size
    k2 = grad(time + 0.5 * step_size, state + 0.5 * k1) * step_size
    k3 = grad(time + 0.5 * step_size, state + 0.5 * k2) * step_size
    k4 = grad(time + step_size, state + k3) * step_size
    return time + step_size, state + (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4_n_steps(grad, time, state, step_size, n_steps):
    """
    Integrates a system of ODEs using the RK4 method over n steps.
    """
    times = np.array([time])
    states = np.array([state])
    for _ in range(n_steps):
        time, state = rk4_step(grad, time, state, step_size)
        times = np.append(times, time)
        states = np.vstack((states, state))
    return times, states

def double_pendulum_grad(time, state, l1, l2, m1, m2, g=9.81):
    """
    Computes the gradient for a double pendulum system given the current state.
    """
    theta1, omega1, theta2, omega2 = state
    
    omega_dot_1=-((g*np.sin(theta1)*m1+g*np.sin(theta1)*m2-g*np.cos(theta1-theta2)*np.sin(theta2)*m2+np.cos(theta1-theta2)*np.sin(theta1-theta2)*l1*m2*omega1**2+np.sin(theta1-theta2)*l2*m2*omega2**2)/(l1*(m1+m2-np.cos(theta1-theta2)**2*m2)))
    omega_dot_2= (g*np.cos(theta1-theta2)*np.sin(theta1)*m1-g*np.sin(theta2)*m1+g*np.cos(theta1-theta2)*np.sin(theta1)*m2-g*np.sin(theta2)*m2+np.sin(theta1-theta2)*l1*m1*omega1**2+np.sin(theta1-theta2)*l1*m2*omega1**2+np.cos(theta1-theta2)*np.sin(theta1-theta2)*l2*m2*omega2**2)/(l2*(m1+m2-np.cos(theta1-theta2)**2*m2))
    
    return np.array([omega1, omega_dot_1, omega2, omega_dot_2])

# Parameters
l1, l2 = 1.0, 1.0  # lengths of the pendulum arms
m1, m2 = 1.0, 1.0 # masses of the pendulum bobs
time_initial = 0
state_initial = np.array([np.pi /2 , 1, np.pi / 1, 0])  # Initial state: [theta1, omega1, theta2, omega2]
step_size = 0.01
n_steps = 10000

# Integration using the RK4 method
times, states = rk4_n_steps(lambda t, s: double_pendulum_grad(t, s, l1, l2, m1, m2), time_initial, state_initial, step_size, n_steps)

def plot_phase_space(states):
    """
    Plots the phase space for each pendulum in a single figure.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(states[:, 0], states[:, 1], label='Pendulum 1')
    plt.xlabel(r"$\theta_1$ (rad)",fontsize=20)
    plt.ylabel(r"$\dot{\theta_1}$ (rad/s)",fontsize=20)
    plt.title('Phase Space for Pendulum 1',fontsize=20)

    plt.subplot(1, 2, 2)
    plt.plot(states[:, 2], states[:, 3], color='pink', label='Pendulum 2')
    plt.xlabel(r"$\theta_2$ (rad)",fontsize=20)
    plt.ylabel(r"$\dot{\theta_2}$ (rad/s)",fontsize=20)
    plt.title('Phase Space for Pendulum 2',fontsize=20)
    
    #params_text = f"Parameters: L1 = {l1}, L2 = {l2}, m1 = {m1}, m2 = {m2}, initial state = [{state_initial[0]:.2f}, {state_initial[1]:.2f}, {state_initial[2]:.2f}, {state_initial[3]:.2f}]"
    #plt.figtext(0.5, -0.1, params_text, ha='center', fontsize=15, wrap=True)
    #plt.tight_layout()
    #plt.show()


    plt.tight_layout()
    plt.show()

# Example usage:
plot_phase_space(states)