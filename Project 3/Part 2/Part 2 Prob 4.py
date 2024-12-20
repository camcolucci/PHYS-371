import numpy as np
from Diff_Class_2D import DiffusionSolver2D

# Case 1: Diffusion of oxygen in air
D_air = 0.18  # Diffusion coefficient for air in cm^2/s
air_solver = DiffusionSolver2D(D=D_air)  # Initialize the solver
print("Running 2D Diffusion Solver for air...")
time_air, snapshots_air = air_solver.solve(t_final=2.0)  # Run the simulation for 2 seconds
air_solver.plot_results(time_air, snapshots_air)  # Plot the results

# Case 2: Diffusion of oxygen in water
D_water = 2e-5  # Diffusion coefficient for water in cm^2/s
water_solver = DiffusionSolver2D(D=D_water)  # Initialize the solver
print("Running 2D Diffusion Solver for water...")
time_water, snapshots_water = water_solver.solve(t_final=10.0)  # Run the simulation for 10 seconds
water_solver.plot_results(time_water, snapshots_water)  # Plot the results

