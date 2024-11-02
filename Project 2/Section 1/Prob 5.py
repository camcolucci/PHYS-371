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
    """Solves the wave equation for a given wave number k using the RK4 method.

    Args:
        L (float): Length of the string.
        N (int): Number of steps for the RK4 solver.
        initial_conditions (list): Initial conditions [y(0), dy/dx(0)].
        k (float): The wave number for the wave equation.

    Returns:
        numpy.ndarray: Solution array with the values of y and dy/dx at each step.
    """
    a = 0  # Start of the interval
    b = L  # End of the interval
    
    # Uses the RK4 solver with the wave system to solve for the system
    rk4_solver = RK4(lambda state, x: First_order_wave_system(state, x, k), a, b, N, initial_conditions)
    
    # Solve the ODEs
    _, solution = rk4_solver.solve()
    return solution

# Function to find the wave number k with tolerance of L/1000
def find_wave_number(L, N, initial_conditions, tolerance=1e-3, max_iter=100):
    """Finds the wave number k that satisfies the boundary condition y(x=L) ≈ 0. This used the Bisection method, which
        is the most straight forward way of approximating a root. The root is considered found when the boundary condition
        is satisfied.

    Args:
        L (float): Length of the string.
        N (int): Number of steps specified for the RK4 solver.
        initial_conditions (list): Initial conditions [y(0), dy/dx(0)].
        tolerance (float): Tolerance for y(x=L) to be considered zero.
        max_iter (int): Maximum number of iterations for the bisection bracket method.

    Returns:
        float: The wave number k that satisfies y(x=L) ≈ 0 within the specified tolerance.
    """
    # Set up initial guesses for k to bracket the solution
    k_low = 1.0  # Starting low guess for k
    k_high = 10.0  # Starting high guess for k

    for i in range(max_iter):
        k_mid = (k_low + k_high) / 2.0  # Midpoint for bisection method
        
        # Solve the system for k_low, k_high, and k_mid using RK4
        y_L_low = solve_wave_equation(L, N, initial_conditions, k_low)[-1, 0]
        y_L_mid = solve_wave_equation(L, N, initial_conditions, k_mid)[-1, 0]
        y_L_high = solve_wave_equation(L, N, initial_conditions, k_high)[-1, 0]
        
        print(f"{i+1}: k = {k_mid}, y(x=L) = {y_L_mid}")
        
        # Check if the midpoint solution is within tolerance at x = L, the loop will break once in tolerance
        if abs(y_L_mid) < tolerance:
            print(f"Solution: k = {k_mid:.3f}, with y(L) = {y_L_mid:.3f}, Wavelength: {(2*np.pi)/k_mid:.3f}")
            return k_mid  # Return k_mid as the final solution
        
        # Uses the bisection method to narrow down the bracket and make the k value more precise
        if y_L_mid * y_L_low < 0:
            k_high = k_mid
        else:
            k_low = k_mid

    raise ValueError(" No Solution found within maximum iterations")

# Main function to initialize and solve the standing wave equation
def main():
    """Main function for solving the standing wave equation on a fixed-open string.
    
    This function sets up the initial conditions and parameters for the problem, including the 
    length of the string, the wavenumber, and initial conditions. It then initializes the 
    RK4 solver, solves the system of differential equations, and plots the results.
    """
    L = 1.0  # Length of the string
    N = 1000  # Number of steps for RK4
    a_init = 1.0  # Initial slope (dy/dx) at x = 0
    initial_conditions = [0, a_init]  # [y(0), dy/dx(0)]
    tolerance = L / 1000  # Tolerance for y(L)
    
    # Find the fundamental wave number k
    k_fundamental = find_wave_number(L, N, initial_conditions, tolerance)
    
    # Solve for the fundamental mode and plot the result
    solution = solve_wave_equation(L, N, initial_conditions, k_fundamental)
    x_points = np.linspace(0, L, N)

    # Plot the result
    plt.plot(x_points, solution[:, 0], label=f"(k = {k_fundamental:.4f})")
    plt.xlabel("x")
    plt.ylabel("Displacement (y)")
    plt.legend()
    plt.title("Standing Wave on a Fixed-Free 1-D System with the Fundamental Mode")
    plt.show()

# Runs the main function to get the results
main()
