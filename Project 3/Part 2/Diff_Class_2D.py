import numpy as np
import matplotlib.pyplot as plt

class DiffusionSolver2D:
    """Solves the 2D diffusion equation using FTCS method."""

    def __init__(self, D, L=2.0, dx=0.05, A=1.0):
        """
        Initialize the Diffusion Solver for a 2D grid.

        Keyword Arguments:
        D -- Thermal Diffusivity in cm^2/s.
        L -- Domain size in cm (default 2.0 cm).
        dx -- Grid spacing in cm (default 0.05 cm).
        A -- Initial concentration value at the center (default 1.0).
        """
        self.D = D  # Diffusion coefficient
        self.L = L  # Length of the simulation domain
        self.dx = dx  # Distance between grid points
        self.A = A  # Initial peak concentration

        # Calculate time step size for stability
        self.dt = (dx**2) / (4 * D) * 0.75  # Stability time step scaled to ensure stability criterion is met
        self.k = D * self.dt / dx**2  # Non-dimensional diffusion constant

        # Stability check for each time step
        if self.k >= 0.25:
            raise ValueError(f"Stability condition not met. k = {self.k} is too large.")

        # Set up the simulation grid
        self.N = int(L / dx)  # Number of grid points per axis
        self.x = np.linspace(-L/2, L/2, self.N)  # x-coordinates for the grid
        self.y = np.linspace(-L/2, L/2, self.N)  # y-coordinates for the grid
        self.X, self.Y = np.meshgrid(self.x, self.y)  # 2D coordinate grid

        # Initialize the concentration arrays
        self.T = np.zeros((self.N, self.N))  # Array for the current concentration
        self.T_updated = np.zeros((self.N, self.N))  # Array for the next step

        # Set the initial condition: peak concentration at the grid center
        center = self.N // 2
        self.T[center, center] = self.A

    def time_step_FTCS(self):
        """
        Perform a single time step update using FTCS.

        Returns:
        max_diff -- The maximum change in concentration during the update.
        """
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                # FTCS update for interior points based on neighbors
                self.T_updated[i, j] = self.T[i, j] + self.k * (
                    self.T[i+1, j] + self.T[i-1, j] +
                    self.T[i, j+1] + self.T[i, j-1] - 4 * self.T[i, j]
                )

        # Calculate the maximum change between time steps
        max_diff = np.max(np.abs(self.T_updated - self.T))

        # Update the main concentration array
        self.T = self.T_updated.copy()

        # Apply boundary conditions: zero concentration at edges
        self.T[0, :] = 0
        self.T[-1, :] = 0
        self.T[:, 0] = 0
        self.T[:, -1] = 0

        return max_diff

    def solve(self, t_final, save_interval=10):
        """
        Solve the diffusion equation until a final time.

        Keyword Arguments:
        t_final -- Total simulation time in seconds.
        save_interval -- Number of equally spaced profiles to save (default 10).

        Returns:
        time -- A list of times when profiles were saved.
        temperature -- A list of 2D concentration arrays corresponding to saved times.
        """
        steps = int(t_final / self.dt)  # Total number of simulation steps
        save_steps = steps // save_interval  # Steps between saving profiles
        time = [0]  # List of times for saved profiles
        temperature = [self.T.copy()]  # Save initial concentration profile

        for n in range(steps):
            diff = self.time_step_FTCS()  # Perform a time step

            # Save profiles at specified intervals
            if n % save_steps == 0:
                time.append((n+1) * self.dt)  # Append the current simulation time
                temperature.append(self.T.copy())  # Save the concentration array

        return time, temperature

    def plot_results(self):
        """
        Plot saved concentration profiles at various times.

        Keyword Arguments:
        time -- List of times corresponding to saved profiles.
        temperature -- List of saved 2D concentration profiles.
        """
        plt.figure(figsize=(8, 6))  # Set figure size for plots
        plt.contourf(self.X, self.Y, T, levels=20, cmap='hot')  # Contour plot of concentrations
        plt.colorbar(label='Concentration')  # Add a colorbar
        plt.title(f"Concentration at t = {t:.2f} s")  # Add a title with time step
        plt.xlabel("x (cm)")  # Label x-axis
        plt.ylabel("y (cm)")  # Label y-axis
        plt.axis('equal')  # Maintain aspect ratio
        plt.show()  # Display the plot

