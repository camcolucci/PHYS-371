import numpy as np
import matplotlib.pyplot as plt

# Constants
l = 1.0   # length of the pendulum in meters
g = 9.81  # gravitational acceleration in m/s^2
theta_0 = np.pi / 4  # initial angle in radians, converts to 45 degrees
omega_0 = 0.0  # initial angular velocity (rad/s), starting from rest
num_oscillations = 10  # number of oscillations to simulate
M = 1.0  # mass of the weight on the pendulum
I = M * l**2  # moment of inertia of the system

# Time parameters
t_0 = 0
T = 2 * np.pi * np.sqrt(l / g)  # period of the pendulum
t_total = num_oscillations * T  # total time for the simulation
step_size = 10000  # number of steps in the simulation
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

# Arrays to store the results
theta_values = np.zeros(step_size)
omega_values = np.zeros(step_size)
time_values = np.linspace(0, t_total, step_size)
E_values = np.zeros(step_size)

# Set initial conditions
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
# value are, or aren't, within the parameter
threshold = 1e-8
outside_threshold = not np.all(np.abs(frac_E_diff) <= threshold)

if outside_threshold:
    max_deviation_index = np.argmax(np.abs(frac_E_diff) > threshold)
    print(f"Fractional Energy did not remain within threshold.")
else:
    print("All Fractional Energy values remained in threshold")

# Create a figure with two y-axes to overlay the oscillation and the fractional energy
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot theta vs time on the first y-axis. This should show the system oscillating
ax1.plot(time_values, theta_values, 'b-', label='Theta (radians)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (radians)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim(-1, 1)

# Create a second y-axis to plot fractional energy difference, this should be a constant line, ideally at 0, to show
# that total energy is fully conserved in the system
ax2 = ax1.twinx()
ax2.plot(time_values, frac_E_diff, 'r-', label='Fractional Energy Difference')
ax2.set_ylabel('Fractional Energy Difference', color='r')
ax2.tick_params('y', colors='r')
ax2.set_ylim(-1, 1)
plt.annotate(f'Fractional Energy Difference: {avg_frac_E_diff:.2e}', 
             xy=(1, 1), xycoords='axes fraction', fontsize=12, color='red',
             xytext=(-10, -10), textcoords='offset points',  # Offset the text a bit from the corner
             ha='right', va='top',  # Align to the top-right
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
plt.title('Nonlinear Pendulum Motion with Energy Conservation')
ax1.grid(True)
plt.show()

"""For this, I just edited the step size and continued to run the code until I got all of the values within the specified
threshold. To ensure all of the values are in, I incorporated the "if" statement and defined the threshold to print a
message stating all of the values ran were in or if some of them deviated. I also included the final value as a part of the
plot so that I could quickly get a rough idea how the value was changing as I adjusted the step size.
    """