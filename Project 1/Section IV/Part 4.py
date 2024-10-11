import numpy as np
import matplotlib.pyplot as plt

a = 0                       # Initial time
b = 10                      # Final time
n_values = [20, 50, 100]    # Iterations
x_i = 1.0                   # Initial x value
v_i = 0.0                   # Initial velocity
omega_0 = 2.0               # Natural Frequency
gamma = omega_0 / 10        # Damping constant
beta = gamma                # Damping coefficient
f_0 = 0.0                   # Driving Force
omega_d = 0.5               # Driving frequency
dt_values = [0.1, 0.4, 0.6, 0.7, 0.8]  # Step sizes for numerical solutions

for n in n_values:          # Setting the number of cycles(iterations) for dt
    t = np.linspace(a, b, n)
    dt = t[1] - t[0]

"""Function that takes inputs omega_0, omega_d and f_0 as parameters to return 
the right side of the damped oscillator equation
"""

def linear_oscillator(x, v, omega_0, omega_d, f_0, t):
    
    dxdt = v   # Defines the first derivative as velocity
    
    dvdt = -2 * beta * v - omega_0**2 * x + f_0 * np.sin(omega_d * t)  # Define the equation of the damped oscillator
                                                                
    return dxdt, dvdt

"""Function that uses the 4th order Runge-Kutta (RK4) to calculate the integral of the equation of motion
    where x_i = x*(t=0) and v_i = v*(t=0)
"""
    
def RK_4th_Order(x_i, v_i, t, dt):
    x_val = np.zeros(len(t))
    v_val = np.zeros(len(t))

 # Initial conditions
    x = x_i
    v = v_i

    for i in range(len(t)):
        x_val[i] = x        # Ranging values for the function
        v_val[i] = v

        # Compute the four different k values using the RK4 Method
        
        k1_x, k1_v = linear_oscillator(x, v, omega_0, omega_d, f_0, t[i])
        
        k2_x, k2_v = linear_oscillator(x + 0.5 * k1_x * dt, v + 0.5 * k1_v * dt, omega_0, omega_d, f_0, t[i] + 0.5 * dt)
        
        k3_x, k3_v = linear_oscillator(x + 0.5 * k2_x * dt, v + 0.5 * k2_v * dt, omega_0, omega_d, f_0, t[i] + 0.5 * dt)
        
        k4_x, k4_v = linear_oscillator(x + k3_x * dt, v + k3_v * dt, omega_0, omega_d, f_0, t[i] + dt)

        # Approximate the position and velocity values using the RK4 formula
    
        x += (k1_x + 2*k2_x + 2*k3_x + k4_x) * dt / 6
        v += (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6

    return x_val, v_val

# Runs the functions and plots the results

x_vals, v_vals = RK_4th_Order(x_i, v_i, t, dt)

plt.plot(t, x_vals, label="Numerical Solution")

# Analytical solution for comparison
def analytical_solution(t, omega_0, beta):
    return np.exp(-beta * t) * np.cos(omega_0 * t)

# Plotting the analytical solution
t_analytical = np.linspace(a, b, 1000)
x_analytical = analytical_solution(t_analytical, omega_0, beta)
plt.plot(t_analytical, x_analytical, color='r', linestyle = '--', label="Analytical Solution")

# Plotting numerical solutions for different delta T values
for dt in dt_values:
    t = np.arange(a, b, dt)  # Time array with variable dt
    x_vals, v_vals = RK_4th_Order(x_i, v_i, t, dt)
    plt.plot(t, x_vals, label=f'Numerical with dt={dt}')

plt.xlabel('Time (s)')
plt.ylabel('Displacement (x)')
plt.legend()
plt.grid(True)
plt.show()