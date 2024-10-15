import numpy as np
import matplotlib.pyplot as plt

# Constants
l = 1.0   # length of the pendulum in meters
g = 9.81  # gravitational acceleration in m/s^2
theta_0 = np.pi / 4  # initial angle in radians, converts to 45 degrees
omega_0 = 0.0  # initial angular velocity (rad/s), starting from rest
num_oscillations = 10  # number of oscillations to simulate

# Time parameters
T = 2 * np.pi * np.sqrt(l / g)  # period of the pendulum
t_total = num_oscillations * T  # total time for the simulation
num_steps = 1000  # number of steps in the simulation
dt = t_total / num_steps  # time step

"""Function for the equations of motion for the nonlinear pendulum
    """
def pendulum_ode(theta, omega):
    d_theta = omega
    d_omega = -(g / l) * np.sin(theta)
    return d_theta, d_omega

"""4th Order RK method used to estimate the motion of the pendulum
    """
def RK_4th_order(theta, omega, dt):
    k1_theta, k1_omega = pendulum_ode(theta, omega)
    k2_theta, k2_omega = pendulum_ode(theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega)
    k3_theta, k3_omega = pendulum_ode(theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega)
    k4_theta, k4_omega = pendulum_ode(theta + dt * k3_theta, omega + dt * k3_omega)

    theta_new = theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_new = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

    return theta_new, omega_new

# Arrays to store the results
theta_values = np.zeros(num_steps)
omega_values = np.zeros(num_steps)
time_values = np.linspace(0, t_total, num_steps)

# Set initial conditions
theta_values[0] = theta_0
omega_values[0] = omega_0

# Time integration loop using RK4
for i in range(1, num_steps):
    theta_values[i], omega_values[i] = RK_4th_order(theta_values[i-1], omega_values[i-1], dt)

# Plot the result: theta vs time
plt.figure(figsize=(10, 6))
plt.plot(time_values, theta_values, label='Theta (radians)')
plt.title('Non Linear Pendulum')
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.grid(True)
plt.show()
