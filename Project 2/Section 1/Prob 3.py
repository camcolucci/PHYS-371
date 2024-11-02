import numpy as np
import matplotlib.pyplot as plt
from FourthOrderRungeKutta import RK4

# Define the system of ODEs for the string wave problem
def First_order_wave_system(state, x, k):
    y, y_prime = state  # state[0] = y, state[1] = dy/dx
    dydx = y_prime
    dy_primedx = -k**2 * y
    return np.array([dydx, dy_primedx])

# Initialize the RK4 solver for the wave equation
def main():
    L = 1.0  # Length (scaled to 1)
    k = 2 * np.pi # Wavenumber for the chosen wavelength (lambda = 2pi/k)
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
