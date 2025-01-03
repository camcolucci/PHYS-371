import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class DiffusionSolver:
    """Diffusion Solver Class
    """
    
    def __init__(self, L=1.0, T_hot=50, T_cold=0, D=0.5, T_init=20, dx=0.01):
        """Initialize the Diffusion Solver

        Args:
        L (float, optional): Length of rod in meters. Defaults to 1.0.
        T_hot (int, optional): Hot reservoir temperature in C. Defaults to 50.
        T_cold (int, optional): Cold reservoir temperature in D. Defaults to 0.
        D (float, optional): Thermal Diffusivity in m^2/s. Defaults to 0.5.
        T_init (int, optional): Initial temperature of the rod in C. Defaults to 20.
        
        """
        # Given parameters
        self.L = L
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.D = D
        self.T_init = T_init
        self.dx = dx
        
        # Derived parameters
        self.N = int(L/dx)  # Number of grid points
        self.dt = dx**2/(2*D) # Time step
        self.k = D*self.dt/dx**2   # Diffusion coefficient
        
        #Stability check of the system
        if self.k > 0.5:
            raise ValueError("Stability condition not met. Reduce time step or increase grid points.")
        
        # Initialize the temperature array
        self.T = np.linspace(0, self.L, self.N) # Temperature array
        self.T = np.full(self.N, T_init)
        self.T_updated = np.zeros(self.N)
        
        #Boundary conditions
        self.T[0] = T_hot
        self.T[-1] = T_cold
        
    def time_step_FTCS(self):
        """Time step using Forward Time Central Space (FTCS) method
        
        Returns:
        float: Maximum temperature difference in the system.
        """
        #Update the temperature array for all interior points
        for i in range(1, self.N-1):
            self.T_updated[i] = self.T[i] + self.k*(self.T[i+1] - 2*self.T[i] + self.T[i-1])
        
        #Implementing boundary conditions
        self.T_updated[0] = self.T_hot
        self.T_updated[-1] = self.T_cold
        
        #Get the maximum temperature difference
        max_diff = np.max(np.abs(self.T_updated - self.T))
        
        #Update the temperature array
        self.T = self.T_updated.copy()
        
        return max_diff
    
    def solve(self, T_f, tolerance=1e-5):
        """Solve the diffusion equation

        Args:
        T_f (float): Final time to solve the equation in seconds.
        tolerance (float, optional): Tolerance for the solution. Defaults to 1e-5.
        
        Returns:
        list: List of time steps.
        list: List of temperature arrays at different time steps.
        """
        steps = int(T_f/self.dt)
        
        #Store the temperature at each time step in a list
        time = [0]
        temperature = [self.T.copy()]
        
        #Time loop
        for i in range(steps):
            diff = self.time_step_FTCS()
            #Store the temperature at each time step
            time.append((i+1)*self.dt)
            temperature.append(self.T.copy())
            #check for steady state
            if diff < tolerance:
                print(f"Steady state reached at time {time[-1]} s.")
                break
        return time, temperature
    
    def plot_results(self, time, temperature):
        """Plot the temperature distribution at different time steps

        Args:
        time (list): List of time steps.
        temperature (list): List of temperature arrays at different time steps.
        
        """
        
        #Create a color map
        color_map = cm.jet(np.linspace(0, 1, len(time)))
        
        # Plot analytical steady state solution
        x = np.linspace(0, self.L, 1000)
        T_ss = -50*x + 50
        plt.plot(x, T_ss, 'k--', label='Analytical steady state')
        
        # Plot selected temperature distributions (e.g., every 12th step) to avoid legend clutter
        step_interval = max(len(time) // 12, 1)  # Adjust step size to show around 10 steps
        for i in range(0, len(time), step_interval):
            plt.plot(np.linspace(0, self.L, self.N), temperature[i],
                 color=color_map[i], label=f"Time: {time[i]:.2f} s")
        
        plt.xlabel("Length (m)")
        plt.ylabel("Temperature (ºC)")
        plt.title("Temperature Distribution Across the Steel Rod")
        plt.legend()
        plt.show()
        
        