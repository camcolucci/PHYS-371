import numpy as np
import matplotlib.pyplot as plt

class RK4:
    def __init__(self, func, a, b, N, x_0, enablePlot=True, runName="4rk"):
        """
        Initializes the RK4 solver.
        
        Parameters:
        func : callable
            The function to approximate (right-hand side of the ODE).
        a : float
            Start of the interval.
        b : float
            End of the interval.
        N : int
            Number of steps.
        x_0 : float
            Initial condition.
        enablePlot : bool
            If True, enables plotting after the solution is computed.
        """
        self.func = func  # The function to approximate (f(x, t))
        self.a = a        # Initial time, or the start of the interval
        self.b = b        # End time, or end of the interval
        self.N = N        # Number of steps
        self.step_size = (b - a) / N  # Step size
        self.x_0 = np.array(x_0)  # Initial condition
        self.enablePlot = enablePlot # Enable plotting of the results
        self.runName = runName

    def solve(self):
        """
        Solves the ODE using the 4th-order Runge-Kutta method (RK4).
        
        Returns:
        tpoints : numpy.ndarray
            Array of time points.
        xpoints : numpy.ndarray
            Array of approximated x values.
        """
        tpoints = np.arange(self.a, self.b, self.step_size)
        xpoints = []
        x = self.x_0
        max_psi_limit = 1e6 

        for t in tpoints:
            xpoints.append(x.copy())  # Append the current state vector

            # Calculate RK4 intermediate steps
            k1 = self.step_size * self.func(x, t)
            k2 = self.step_size * self.func(x + 0.5 * k1, t + 0.5 * self.step_size)
            k3 = self.step_size * self.func(x + 0.5 * k2, t + 0.5 * self.step_size)
            k4 = self.step_size * self.func(x + k3, t + self.step_size)

            # Update the solution
            x += (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Check for divergence
            if np.any(np.abs(x) > max_psi_limit):  # If any element exceeds the limit
                print("Warning: Solution diverged at t =", t)
                break
        return np.array(tpoints[:len(xpoints)]), np.array(xpoints)

    def plot(self, tpoints, xpoints):
        """
        Plots the results of the ODE approximation.
        
        Parameters:
        tpoints : numpy.ndarray
            Array of time points.
        xpoints : numpy.ndarray
            Array of approximated x values.
        """
        plt.plot(tpoints, xpoints[:, 0], label='Displacement')
        plt.plot(tpoints, xpoints[:, 1], label='Velocity', linestyle='--')
        plt.xlabel("Position $x$")
        plt.ylabel("Displacement and Velocity")
        plt.legend()
        if self.enablePlot:
            plt.show()


