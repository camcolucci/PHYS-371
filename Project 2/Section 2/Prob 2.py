import numpy as np
from scipy.integrate import quad

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

# Displays the results and labels which terms are non zero
for index, (n_value, coeff) in enumerate(non_zero_coefficients, start=1):
    print(f"C_{n_value} = {coeff:.5f}")

