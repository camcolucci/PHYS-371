import numpy as np
import matplotlib.pyplot as plt

# Initial conditions and parameters
xi = 1.0             # Initial position
vi = 0.0             # Initial velocity
a = 0.0              # Start time
b = 100.0            # End time
omega_0 = 2.0        # Natural frequency
f_0 = 50.0           # Driving force amplitude
dt = 0.01            # Time step
beta = omega_0 / 10  # Damping coefficient
omega_d = np.linspace(0.5, 3.0, 500)  # Driving frequencies

"""Function that takes inputs omega_0, omega_d and f_0 as parameters to return 
the right side of the damped oscillator equation (acceleration)
"""
def linear_oscillator(t, x, v, omega0, beta, f0, omega_d_val):  # Calculate acceleration
    return -2 * beta * v - omega0**2 * x + f0 * np.sin(omega_d_val * t)

"""Function that uses the 4th order Runge-Kutta (RK4) to calculate the integral of the equation of motion
    where x_i = x*(t=0) and v_i = v*(t=0)
"""
def RK_4th_Order(xi, vi, t0, tf, dt, omega0, beta, f0, omega_d_val): 
    t = np.arange(t0, tf, dt)         # Time
    x_values = np.zeros(len(t))       # Position
    v_values = np.zeros(len(t))       # Velocity
    x, v = xi, vi                     # Initial position and velocity
    
    for i in range(len(t)):  # Time loop through x and v
        x_values[i] = x      
        v_values[i] = v           
        
        # Compute the four different k values using the RK4 Method
        k1_x = v
        k1_v = linear_oscillator(t[i], x, v, omega0, beta, f0, omega_d_val)
        
        k2_x = v + 0.5 * dt * k1_v
        k2_v = linear_oscillator(t[i] + 0.5 * dt, x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v, omega0, beta, f0, omega_d_val)
        
        k3_x = v + 0.5 * dt * k2_v
        k3_v = linear_oscillator(t[i] + 0.5 * dt, x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v, omega0, beta, f0, omega_d_val)
        
        k4_x = v + dt * k3_v
        k4_v = linear_oscillator(t[i] + dt, x + dt * k3_x, v + dt * k3_v, omega0, beta, f0, omega_d_val)
        
        # Approximate the position and velocity values using the RK4 formula
        x += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    
    return t, x_values, v_values

# Calculate amplitudes as a function of driving frequency
amplitudes = []
for omega_d_val in omega_d: 
    t, x_numerical, _ = RK_4th_Order(xi, vi, a, b, dt, omega_0, beta, f_0, omega_d_val)
    steady_state_amplitude = np.max(np.abs(x_numerical[int(0.8 * len(t)):]))
    amplitudes.append(steady_state_amplitude)

plt.figure(figsize=(10, 6))
plt.plot(omega_d, amplitudes, label=r'Amplitude vs Driving Frequency')
plt.title(r'Frequency Dependent Amplitude A($\omega_d$)')
plt.xlabel(r'Driving Frequency $\omega_d$')
plt.ylabel(r'Amplitude A($\omega_d$)')
plt.grid(True)
plt.legend(loc='best')
plt.show()

