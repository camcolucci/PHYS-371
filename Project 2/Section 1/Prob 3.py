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

# Initializes the RK4 solver for the wave equation
def main():
    """Main function for solving the standing wave equation on a fixed-open string.
    
    This function sets up the initial conditions and parameters for the problem, including the 
    length of the string, the wavenumber, and initial conditions. It then initializes the 
    RK4 solver, solves the system of differential equations, and plots the results.
    """
    L = 1.0  # Length
    Lam = 1.0
    k = (2 * np.pi) / Lam # Wavenumber for the chosen wavelength (lambda = 2pi/k)
    N = 1000  # Number of steps
    a = 0  # Start of the interval
    b = L  # End of the interval
    
    # Initial conditions y(0) = 0, dy/dx(0) = a
    a_init = 1.0  # Amplitude of dy/dx at x = 0
    initial_conditions = [0, a_init]  # [y(0), dy/dx(0)]
    
    # Create an instance of the RK4 solver with the wave system
    rk4_solver = RK4(lambda state, x: First_order_wave_system(state, x, k), a, b, N, initial_conditions)
    
    # Solve the ODEs
    x_points, solution = rk4_solver.solve()
    
    # Plot the results
    rk4_solver.plot(x_points, solution)

# Run the main function for the result
main()
