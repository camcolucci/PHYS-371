import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parameters
L = 1.0  # Length of the string
c = 1.0  # Wave speed
num_snapshots = 8  # Number of snapshots
time_values = np.linspace(0, 2 * np.pi / c, num_snapshots)  # Times within one period

# Initial conditions of the piecewise function ξ(x, t=0)
def xi_initial(x):
    if x < L / 3:
        return 3 * x / L
    elif L / 3 <= x <= 2 * L / 3:
        return 1
    elif 2 * L / 3 < x <= L:
        return -3 * x / L + 3
    else:
        return 0

# Function to calculate C_n
def C_n(n):
    # Set up the integral
    integrand = lambda x: xi_initial(x) * np.sin(n * np.pi * x / L)
    # Integrate from 0 to L
    result, _ = quad(integrand, 0, L)
    # Multiply by the normalization factor of 2/L
    return (2 / L) * result

# Find the first 10 non zero C_n terms
non_zero_coefficients = []
n = 1  # Initialize at the first term
tolerance = 1e-6  # Tolerance to consider values as non-zero, these will be skipped

while len(non_zero_coefficients) < 10:
    coefficient = C_n(n)
    if abs(coefficient) > tolerance:
        non_zero_coefficients.append((n, coefficient))
    n += 1

# Define x values for the partial sums
x_values = np.linspace(0, L, 500)

# Plot the snapshots
plt.figure(figsize=(10, 8))
for t in time_values:
    # Compute the solution at time t using the first 10 non-zero terms of C_n
    xi_values = np.zeros_like(x_values)
    for n_value, coeff in non_zero_coefficients:
        k_n = n_value * np.pi / L
        omega_n = c * k_n
        xi_values += coeff * np.sin(k_n * x_values) * np.cos(omega_n * t)
    
    plt.plot(x_values, xi_values, label=f't = {t:.2f} s')

plt.xlabel('Displacement (x)')
plt.ylabel(r'$\xi(x, t)$')
plt.title('Time Snapshots of the String')
plt.legend()
plt.grid(True)
plt.show()
