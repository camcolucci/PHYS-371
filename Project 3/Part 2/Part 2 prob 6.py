import numpy as np
import matplotlib.pyplot as plt
from Diff_Class_2D import DiffusionSolver2D

# Function to compute average position
def compute_avg_position(temperature, axis, coordinates):
    avg_position = []
    for T in temperature:
        avg = np.sum(T * coordinates) / np.sum(T)
        avg_position.append(avg)
    return avg_position

# Case 1: Diffusion of oxygen in air
air_solver = DiffusionSolver2D(D=0.18, L=2.0, dx=0.05, A=1.0)
times_air, snapshots_air = air_solver.solve(t_final=2.0, save_interval=10)
msd_air = air_solver.compute_msd(snapshots_air)
avg_x_air = compute_avg_position(snapshots_air, axis=1, coordinates=air_solver.x)
avg_y_air = compute_avg_position(snapshots_air, axis=0, coordinates=air_solver.y)

# Plot Average Position vs Time for Air
plt.figure(figsize=(8, 6))
plt.plot(times_air, avg_x_air, label="⟨x⟩", marker='o', linestyle='-')
plt.plot(times_air, avg_y_air, label="⟨y⟩", marker='s', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Average Position vs Time (Air)")
plt.legend()
plt.grid(True)
plt.show()

# Plot MSD vs Time for Air
plt.figure(figsize=(8, 6))
plt.plot(times_air, msd_air, marker='o', linestyle='-', color='b')
plt.xlabel("Time (s)")
plt.ylabel("Mean Squared Displacement (cm²)")
plt.title("MSD vs Time (Air)")
plt.grid(True)
plt.show()

# Case 2: Diffusion of oxygen in water
water_solver = DiffusionSolver2D(D=2e-5, L=2.0, dx=0.05, A=1.0)
times_water, snapshots_water = water_solver.solve(t_final=1000.0, save_interval=10)
msd_water = water_solver.compute_msd(snapshots_water)
avg_x_water = compute_avg_position(snapshots_water, axis=1, coordinates=water_solver.x)
avg_y_water = compute_avg_position(snapshots_water, axis=0, coordinates=water_solver.y)

# Plot Average Position vs Time for Water
plt.figure(figsize=(8, 6))
plt.plot(times_water, avg_x_water, label="⟨x⟩", marker='o', linestyle='-')
plt.plot(times_water, avg_y_water, label="⟨y⟩", marker='s', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Average Position vs Time (Water)")
plt.legend()
plt.grid(True)
plt.show()

# Plot MSD vs Time for Water
plt.figure(figsize=(8, 6))
plt.plot(times_water, msd_water, marker='s', linestyle='-', color='r')
plt.xlabel("Time (s)")
plt.ylabel("Mean Squared Displacement (cm²)")
plt.title("MSD vs Time (Water)")
plt.grid(True)
plt.show()

