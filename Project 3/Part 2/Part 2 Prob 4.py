import numpy as np
from Diff_Class_2D import DiffusionSolver2D

# Initialize the solver for Case #1
air_solver = DiffusionSolver2D(D=0.18)

print("Running 2D Diffusion Solver for air...")
times, snapshots = air_solver.solve(t_final=2.0)
air_solver.plot_results()

#Initialize solver for Case #2
water_solver = DiffusionSolver2D(D=2e-5)

print("Running 2D Diffusion Solver for water...")
times, snapshots = water_solver.solve(t_final=10.0)
water_solver.plot_results()
