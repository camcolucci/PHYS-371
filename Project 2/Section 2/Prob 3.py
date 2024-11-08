import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

L = 1.0  # Length of the string

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

# Define x values for the partial sum
x_values = np.linspace(0, L, 500)

# Plot partial sums overlayed with each other
plt.figure(figsize=(10, 8))
for k in range(1, 11):
    partial_sum = np.zeros_like(x_values)
    # Calculate the partial sum for the first 10 non zero C_n values
    for j in range(k):
        n_value, coeff = non_zero_coefficients[j]
        partial_sum += coeff * np.sin(n_value * np.pi * x_values / L)
    
    plt.plot(x_values, partial_sum, label=f'Term {k}')

# Plot the exact initial shape to compare the partial sums to
exact_shape = [xi_initial(x) for x in x_values]
plt.plot(x_values, exact_shape, 'k--', label=r'$\xi(x, t=0)$')

plt.xlabel('Displacement (x)')
plt.ylabel(r'$\xi(x, t=0)$')
plt.title('Partial Sum and Exact Value Comparison of Fourier Coefficients')
plt.legend()
plt.grid(True)
plt.show()

