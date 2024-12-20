import numpy as np
import matplotlib.pyplot as plt

class DiffusionSolver2D:
    """Solves the 2D diffusion equation using FTCS method."""

    def __init__(self, D, L=2.0, dx=0.05, A=1.0):
        """
        Initialize the Diffusion Solver for a 2D grid.

        Keyword Arguments:
        D -- Diffusion coefficient in cm^2/s.
        L -- Domain size in cm (default 2.0 cm).
        dx -- Grid spacing in cm (default 0.05 cm).
        A -- Initial concentration value at the center (default 1.0).
        """
        self.D = D  # Diffusion coefficient in cm^2/s
        self.L = L  # Length of the simulation domain in cm
        self.dx = dx  # Distance between grid points in cm
        self.A = A  # Initial peak concentration at the grid center

        # Calculate time step size for stability and scale it for safety
        self.dt = (dx**2) / (4 * D) * 0.9  # Stability time step scaled by 0.9
        self.k = D * self.dt / dx**2  # Non-dimensional diffusion constant

        # Stability check for the time step
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

    def compute_msd(self, temperature):
        """
        Compute the Mean Squared Displacement (MSD) at each saved time step.

        Keyword Arguments:
        temperature -- List of saved 2D concentration profiles.

        Returns:
        msd -- List of MSD values for each time step.
        """
        msd = []  # Initialize list to store MSD values
        for T in temperature:
            # Create meshgrid for squared distances
            x_squared = self.X**2
            y_squared = self.Y**2

            # Compute MSD: \langle r^2 \rangle = \sum c(x, y) * (x^2 + y^2) / \sum c(x, y)
            msd_value = np.sum(T * (x_squared + y_squared)) / np.sum(T)
            msd.append(msd_value)

        return msd

    def plot_msd(self, time, msd):
        """
        Plot Mean Squared Displacement (MSD) as a function of time.

        Keyword Arguments:
        time -- List of times corresponding to saved profiles.
        msd -- List of MSD values for each time step.
        """
        plt.figure(figsize=(9, 6))  # Set figure size for plots
        plt.plot(time, msd, marker='o', linestyle='-', color='b')  # MSD vs. time plot
        plt.xlabel("Time (s)")  # Label x-axis
        plt.ylabel("Mean Squared Displacement (cm^2)")  # Label y-axis
        plt.title("Mean Squared Displacement vs Time")  # Title
        plt.grid(True)  # Add grid lines
        plt.show()  # Display the plot

    def plot_results(self, time, temperature, num_plots=4):
        """
        Plot saved concentration profiles at various times in a grid layout.

        Keyword Arguments:
        time -- List of times corresponding to saved profiles.
        temperature -- List of saved 2D concentration profiles.
        num_plots -- Number of profiles to display (default 4).
        """
        # Select evenly spaced profiles for time plots
        indices = np.linspace(0, len(time) - 1, num_plots, dtype=int)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Create 2x2 subplot grid
        axes = axes.ravel()  # Flatten axes array for easy iteration

        for idx, ax in zip(indices, axes):
            # Plot the concentration profile for the selected time
            im = ax.contourf(self.X, self.Y, temperature[idx], levels=20, cmap='hot')
            ax.set_title(f't = {time[idx]:.3f} s')  # Add title with time
            ax.set_xlabel('x (cm)')  # Label x-axis
            ax.set_ylabel('y (cm)')  # Label y-axis
            ax.set_aspect('equal')  # Maintain equal aspect ratio

            # Add colorbar for each plot
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Concentration')

        plt.suptitle(rf'Diffusion Evolution (D = {self.D} $cm^2/s$)', fontsize=16)  # Super title
        plt.tight_layout()  # Adjust layout
        plt.show()  # Display the plots



