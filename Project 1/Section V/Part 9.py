import numpy as np
import matplotlib.pyplot as plt

# Constants
l = 1.0   # length of the pendulum in meters
g = 9.81  # gravitational acceleration in m/s^2
theta_0 = np.pi / 4  # initial angle in radians, converts to 45 degrees
omega_0 = 0.0  # initial angular velocity (rad/s), starting from rest
num_oscillations = 2  # number of oscillations to simulate
M = 1.0  # mass of the weight on the pendulum
I = M * l**2  # moment of inertia of the system

# Time parameters
t_0 = 0
T = 2 * np.pi * np.sqrt(l / g)  # period of the pendulum
t_total = num_oscillations * T  # total time for the simulation
step_size = 1000  # number of steps in the simulation
dt = t_total / step_size  # time step

# Function for the equations of motion for the nonlinear pendulum
def pendulum_ode(state):
    theta, omega = state
    d_theta = omega
    d_omega = -(g / l) * np.sin(theta)
    return np.array([d_theta, d_omega])

# 4th Order Runge-Kutta method
def RK_4th_order(state, dt):
    k1 = dt * pendulum_ode(state)
    k2 = dt * pendulum_ode(state + 0.5 * k1)
    k3 = dt * pendulum_ode(state + 0.5 * k2)
    k4 = dt * pendulum_ode(state + k3)
    return state + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Computes the total energy of the system
def tot_E(I, omega, M, theta, g, l):
    KE = 0.5 * I * omega**2
    PE = M * g * l * (1 - np.cos(theta))
    return KE + PE

# Arrays to store the results for the initial theta_0 value
theta_values = np.zeros(step_size)
omega_values = np.zeros(step_size)
time_values = np.linspace(0, t_total, step_size)
E_values = np.zeros(step_size)

# Set initial conditions for the original theta_0
state = np.array([theta_0, omega_0])
theta_values[0] = state[0]
omega_values[0] = state[1]
E_values[0] = tot_E(I, state[1], M, state[0], g, l)

# Time integration loop using RK4 and the specified step size
for i in range(1, step_size):
    state = RK_4th_order(state, dt)
    theta_values[i], omega_values[i] = state
    E_values[i] = tot_E(I, omega_values[i], M, theta_values[i], g, l)

# Calculate initial energy E(t=0)
E_0 = E_values[0]

# Compute fractional energy difference. The average is computed just for a value to show on the plot so that I can
# see if things are going in the right direction
frac_E_diff = (E_values - E_0) / E_0
avg_frac_E_diff = np.mean(frac_E_diff)

# Check if the energy deviation is within the threshold. A message will print that says whether all of the 
# values are, or aren't, within the parameter
threshold = 1e-8
outside_threshold = not np.all(np.abs(frac_E_diff) <= threshold)

if outside_threshold:
    max_deviation_index = np.argmax(np.abs(frac_E_diff) > threshold)
    print(f"Fractional Energy did not remain within threshold.")
else:
    print("All Fractional Energy values remained in threshold")

# Function to compute the small-angle solution for different initial angles
def small_theta_sol(theta_0, g, l, t_values):
    return theta_0 * np.cos(np.sqrt(g / l) * t_values)

# List of small initial angles for which we want to calculate the analytic and numerical solutions
small_angles = [np.pi / 8, np.pi / 16, np.pi / 24, np.pi / 30, np.pi / 36]

# Create a figure with two y-axes to overlay the oscillation and the fractional energy
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the numerical solution for theta (oscillations) on the first y-axis for each small angle
for theta_0 in small_angles:
    # Arrays to store the results for the numerical solution
    theta_values = np.zeros(step_size)
    omega_values = np.zeros(step_size)
    E_values = np.zeros(step_size)

    # Set initial conditions for the numerical solution for each theta_0
    state = np.array([theta_0, omega_0])
    theta_values[0] = state[0]
    omega_values[0] = state[1]
    E_values[0] = tot_E(I, state[1], M, state[0], g, l)

    # Time integration loop for each small angle
    for i in range(1, step_size):
        state = RK_4th_order(state, dt)
        theta_values[i], omega_values[i] = state
        E_values[i] = tot_E(I, omega_values[i], M, theta_values[i], g, l)

    # Compute the small-angle analytic solution for each theta_0
    theta_analytic = small_theta_sol(theta_0, g, l, time_values)

    # Plot both the numerical solution and the analytic solution
    ax1.plot(time_values, theta_values, 'b-', alpha=0.6, label=f'Numerical Solution ($\\theta_0={theta_0:.2f}$)')
    ax1.plot(time_values, theta_analytic, '--', label=f'Analytic Solution ($\\theta_0={theta_0:.2f}$)')


ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (radians)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim(-0.45, 0.45)
ax1.legend(loc='upper left')

# Create a second y-axis to plot fractional energy difference
# ax2 = ax1.twinx()
# ax2.plot(time_values, frac_E_diff, 'r-', label='Fractional Energy Difference')
# ax2.set_ylabel('Fractional Energy Difference', color='r')
# ax2.tick_params('y', colors='r')
# ax2.set_ylim(-1, 1)
# ax2.legend(loc='upper right')

plt.title('Nonlinear Pendulum Motion with Small-Angle Comparison')
ax1.grid(True)
plt.show()

""" "Agrees well" means that the numerical and analytical solutions have minimal deviation, 
    especially in terms of amplitude and period. For small initial angles, the RMSE is low, 
    indicating good agreement, as the small-angle approximation holds true. For larger angles, 
    the deviation becomes more noticeable due to the nonlinear effects not captured by the 
    small-angle approximation.
    """
