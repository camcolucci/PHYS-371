#%%
import numpy as np
import matplotlib.pyplot as plt

# Initialize the variables that will define the function.

a = 0                          # Starting point
b = 20                         # Ending point
n = 1000                       # Number of iterations, must be even
m = 1                          # Mass of the block in the oscillator
k = 1                          # Spring constant
B = 1                          # Damping Factor
beta = B / (2*m)               # Damping Coefficient
omega_0 = np.sqrt(k / m)       # Natural Frequency
t = np.linspace(a, b, n)       # Creates time values for the function
alpha = 0.1                    # Decay constant of the driving force
dt = t[1] - t[0]               # Change in time/step size

# Define the force function F(t')

def F(t_prime):                
    return m * np.exp(-alpha * t_prime)
  
# Ensures that n is even for Simpson's rule
  
def subintervals(n):          
    if n % 2 == 1:
        n -= 1
    return n

# Define Green's function G(t-t')

def G(t, t_prime):
    tau = t - t_prime
    if tau < 0:
        return 0
    return (1 / (m * omega_0)) * np.exp(-beta * tau) * np.sin(omega_0 * tau)

# Set up Simpson's rule formula

def simpsons_rule(functions, h):
    n = len(functions)
    result = functions[0] + functions[-1]       # First and last terms
    result += 4 * np.sum(functions[1:n:2])      # Odd terms
    result += 2 * np.sum(functions[2:n-1:2])    # Even terms
    return h/3 * result

# Find the total displacement x(t)

def compute_SHO(t):
    t_prime_value = np.linspace(a, t, n)                                            # Time points for integration
    calculated_value = [F(t_prime) * G(t, t_prime) for t_prime in t_prime_value]    # Integrand values
    return simpsons_rule(calculated_value, dt)                                      # Use Simpson's rule to integrate

# Compute x(t) over the entire time range

x_t = np.array([compute_SHO(ti) for ti in t])

# Plots the result as a smooth curve, showing that this is a smooth potential energy function

plt.plot(t, x_t,)
plt.xlabel("Time (t)")
plt.ylabel("Displacement of the mass x(t)")
plt.title("Smooth Potential Energy Function (Harmonic Oscillator)")
plt.grid(True)
plt.legend()
plt.show()

# %%