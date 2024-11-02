import numpy as np
import matplotlib.pyplot as plt
from FourthOrderRungeKutta import RK4

# Define the system of ODEs for the string wave problem
def First_order_wave_system(state, x, k):
    """Defines the first-order system of differential equations for a standing wave.

    Args:
        state (numpy.ndarray): A 2-element array where state[0] is the displacement y,
                               and state[1] is the velocity dy/dx.
        x (float): The position along the string, used as the independent variable in RK4.
        k (float): The wavenumber, related to the wavelength by k = 2*pi / lambda.

    Returns:
        numpy.ndarray: A 2-element array containing [dy/dx, d^2y/dx^2] evaluated at the given state.
                       dy/dx is the first derivative of y with respect to x,
                       and d^2y/dx^2 is derived from the wave equation.
    """
    y, y_prime = state  # y = state[0], dy/dx = state[1]
    dydx = y_prime
    dy_prime_dx = -k**2 * y
    return np.array([dydx, dy_prime_dx])

# Function to solve the wave equation for a specific wave number k
def solve_wave_equation(L, N, initial_conditions, k):
    """Solves the wave equation for a given wave number k using the RK4 method."""
    a = 0  # Start of the interval
    b = L  # End of the interval
    
    # Uses the RK4 solver with the wave system to solve for the system
    rk4_solver = RK4(lambda state, x: First_order_wave_system(state, x, k), a, b, N, initial_conditions)
    
    # Solve the ODEs
    _, solution = rk4_solver.solve()
    return solution

# Function to find the wave number k with tolerance of L/1000
def find_wave_number(L, N, initial_conditions, tolerance=1e-3, max_iter=100, k_initial=None):
    """Finds the wave number k that satisfies the boundary condition y(x=L) ≈ 0 using the Bisection method."""
    k_low = k_initial * 0.9 if k_initial else 1.0  # Starting low guess for k
    k_high = k_initial * 1.1 if k_initial else 10.0  # Starting high guess for k

    for i in range(max_iter):
        k_mid = (k_low + k_high) / 2.0
        y_L_mid = solve_wave_equation(L, N, initial_conditions, k_mid)[-1, 0]
        
        if abs(y_L_mid) < tolerance:
            return k_mid
        
        y_L_low = solve_wave_equation(L, N, initial_conditions, k_low)[-1, 0]
        if y_L_mid * y_L_low < 0:
            k_high = k_mid
        else:
            k_low = k_mid

    raise ValueError("No solution found within maximum iterations")

# Function to find wave numbers for modes n=1, n=2, and n=3
def find_harmonic_wave_numbers(L, N, initial_conditions, tolerance=1e-3):
    """Finds the fundamental and first two harmonic wave numbers."""
    results = {}
    for n in range(1, 4):
        k_initial = n * np.pi / L  # Approximate wave number for the nth mode
        k_n = find_wave_number(L, N, initial_conditions, tolerance, k_initial=k_initial)
        results[f"n={n}"] = k_n
    return results

# Main function to initialize and solve the standing wave equation
def main():
    """Main function for solving and plotting the standing wave equation for modes n=1, 2, and 3 on a fixed-open string."""
    L = 1.0  # Length of the string
    N = 100  # Number of steps for RK4
    a_init = 1.0  # Initial slope (dy/dx) at x = 0
    initial_conditions = [0, a_init]  # [y(0), dy/dx(0)]
    tolerance = L / 1000  # Tolerance for y(L)

    # Find the wave numbers for the fundamental and first two harmonics
    harmonic_wave_numbers = find_harmonic_wave_numbers(L, N, initial_conditions, tolerance)
    
    # x values for plotting
    x_points = np.linspace(0, L, N)
    
    # Loop through modes n=1, n=2, and n=3, solve and plot each, and print results
    for n in range(1, 4):
        numerical_k = harmonic_wave_numbers[f"n={n}"]
        solution = solve_wave_equation(L, N, initial_conditions, numerical_k)
        
        # Analytical values for the current mode
        analytic_k = n * np.pi / L
        analytic_lambda = 2 * np.pi / analytic_k
        analytic_frequency = analytic_k / (2 * np.pi)  # Assuming wave speed v = 1

        # Numerical values derived from the computed wave number
        numerical_lambda = 2 * np.pi / numerical_k
        numerical_frequency = numerical_k / (2 * np.pi)

        # Print the values for comparison
        print(f"\nMode n={n}:")
        print(f"  Numerical k = {numerical_k:.6f}, Analytical k = {analytic_k:.6f}, "
              f"Fractional Discrepancy = {abs(numerical_k - analytic_k) / analytic_k:.6f}")
        print(f"  Numerical wavelength = {numerical_lambda:.6f}, Analytical wavelength = {analytic_lambda:.6f}, "
              f"Fractional Discrepancy = {abs(numerical_lambda - analytic_lambda) / analytic_lambda:.6f}")
        print(f"  Numerical frequency = {numerical_frequency:.6f}, Analytical frequency = {analytic_frequency:.6f}, "
              f"Fractional Discrepancy = {abs(numerical_frequency - analytic_frequency) / analytic_frequency:.6f}")

        # Plot the displacement for each mode
        plt.plot(x_points, solution[:, 0], label=f"Mode n={n} (k = {numerical_k:.4f})")
    
    # Configure the plot
    plt.xlabel("x")
    plt.ylabel("Displacement y")
    plt.legend()
    plt.title("Standing Wave on a Fixed-Free 1-D String for Modes n=1, 2, and 3")
    plt.show()

# Runs the main function to get the results
main()

