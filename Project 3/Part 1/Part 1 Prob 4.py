import numpy as np
from Diffusion_Class import DiffusionSolver

L = 1  # Length of the rod in meters
D = 1.172e-5  # Thermal diffusivity of 1010 steel in m^2/s
dx = 0.05  # Spatial step size in meters
t_f = 10000.0  # Final time in seconds

# Create the diffusion solver object
solver = DiffusionSolver(L=L, D=D, dx=dx, T_cold=0.0, T_hot=50.0, T_init=20.0)

# Solve the diffusion equation
time, temperatures = solver.solve(t_f)

# Plot the results in a temperature vs. position plot
solver.plot_results(time, temperatures)

# Build and plot the analytical solution in steady state
x = np.linspace(0, L, solver.N)  # Spatial grid
T_analytical = 50 - 50 * x / L  # Steady-state analytical solution

# Compare the analytical solution with the numerical solution
numerical_solution = temperatures[-1]  # Steady-state numerical solution
error = np.max(np.abs(numerical_solution - T_analytical))
print(f"Max error between the analytical and numerical solution: {error:.4f} ºC")














