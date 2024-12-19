import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class DiffusionSolver_2D:
    """2D Diffusion Solver Class
    Solves the 2D heat diffusion equation using the FTCS (Forward Time Central Space) method.
    """
    
    def __init__(self, L=1.0, T_hot=50, T_cold=0, D, T_init=20, dx=0.01):
        """Initialize the 2D Diffusion Solver

        Args:
        L (float, optional): Length of the square domain in meters. Defaults to 1.0.
        T_hot (int, optional): Temperature at the hot boundaries in °C. Defaults to 50.
        T_cold (int, optional): Temperature at the cold boundaries in °C. Defaults to 0.
        D (float, optional): Thermal diffusivity in cm²/s.
        T_init (int, optional): Initial temperature of the domain in °C. Defaults to 20.
        dx (float, optional): Spatial step size in meters. Defaults to 0.01.
        """
        # Given parameters
        self.L = L
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.D = D
        self.T_init = T_init
        self.dx = dx
        
        # Derived parameters
        self.N = int(L / dx)  # Number of grid points in each direction
        self.dt = dx**2 / (4 * D)  # Time step size based on stability condition
        self.k = D * self.dt / dx**2  # Diffusion coefficient

        # Stability check for FTCS scheme
        if self.k > 0.25:
            raise ValueError("Stability condition not met. Reduce time step or increase grid points.")

        # Initialize the temperature array
        self.T = np.full((self.N, self.N), T_init)  # 2D array initialized to T_init
        self.T_updated = np.zeros_like(self.T)  # Placeholder for the updated temperature array
        
        # Apply boundary conditions
        self.T[:, 0] = T_hot   # Left boundary
        self.T[:, -1] = T_cold  # Right boundary
        self.T[0, :] = T_hot   # Top boundary
        self.T[-1, :] = T_cold  # Bottom boundary
    
    def time_step_FTCS(self):
        """Perform one time step using the Forward Time Central Space (FTCS) method.
        
        Updates the temperature field based on the central difference in both spatial directions.

        Returns:
        float: Maximum temperature difference in the system after the update.
        """
        # Update the temperature array for all interior points
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                self.T_updated[i, j] = self.T[i, j] + self.k * (
                    self.T[i + 1, j] + self.T[i - 1, j] + self.T[i, j + 1] + self.T[i, j - 1] - 4 * self.T[i, j]
                )
        
        # Reapply boundary conditions
        self.T_updated[:, 0] = self.T_hot   # Left boundary
        self.T_updated[:, -1] = self.T_cold  # Right boundary
        self.T_updated[0, :] = self.T_hot   # Top boundary
        self.T_updated[-1, :] = self.T_cold  # Bottom boundary

        # Calculate the maximum temperature difference
        max_diff = np.max(np.abs(self.T_updated - self.T))

        # Update the temperature field
        self.T[:] = self.T_updated
        return max_diff
    
    def solve(self, T_f, tolerance=1e-5):
        """Solve the 2D diffusion equation over time.

        Args:
        T_f (float): Final simulation time in seconds.
        tolerance (float, optional): Tolerance for convergence to steady state. Defaults to 1e-5.

        Returns:
        list: List of time steps.
        list: List of temperature fields at different time steps.
        """
        steps = int(T_f / self.dt)  # Total number of time steps
        time = [0]  # Time step storage
        temperature = [self.T.copy()]  # Store the initial temperature field
        
        # Time loop
        for step in range(steps):
            diff = self.time_step_FTCS()  # Perform a single time step
            
            # Store results at each step
            time.append((step + 1) * self.dt)
            temperature.append(self.T.copy())
            
            # Check for steady state
            if diff < tolerance:
                print(f"Steady state reached at time {time[-1]:.2f} s.")
                break
        
        return time, temperature
    
    def plot_results(self, time, temperature):
        """Plot the temperature distribution as heatmaps at selected time steps.

        Args:
        time (list): List of time steps.
        temperature (list): List of temperature fields at different time steps.
        """
        step_interval = max(len(time) // 4, 1)  # Select 4 time steps for visualization
        plt.figure(figsize=(10, 6))
        
        # Plot temperature distribution at selected time steps
        for idx in range(0, len(time), step_interval):
            plt.imshow(temperature[idx], cmap=cm.jet, origin="lower", extent=[0, self.L, 0, self.L],
                       vmin=self.T_cold, vmax=self.T_hot)
            plt.colorbar(label="Temperature (°C)")
            plt.title(f"Temperature Distribution at t = {time[idx]:.2f} s")
            plt.xlabel("X Position (cm)")
            plt.ylabel("Y Position (cm)")
            plt.show()


        