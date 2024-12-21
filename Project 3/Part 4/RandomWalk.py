import numpy as np
import matplotlib.pyplot as plt

class RandomWalkSimulator:
    """
    A simulator for 2D random walks involving multiple particles.

    The simulation involves a specified number of particles performing random walks with a fixed step size 
    and random direction for each step.
    """

    def __init__(self, particle_count=1000, step_count=10000, step_size=1.0):
        """
        Initialize the random walk simulator.

        Parameters:
        ----------
        particle_count : int, optional
            Number of particles to simulate. Default is 1000.
        step_count : int, optional
            Number of steps per particle. Default is 10000.
        step_size : float, optional
            Length of each step. Default is 1.0.
        """
        self.particle_count = particle_count
        self.step_count = step_count
        self.step_size = step_size

        # Initialize particle positions and random angle storage
        self.positions_x = np.zeros((particle_count, step_count + 1))
        self.positions_y = np.zeros((particle_count, step_count + 1))
        self.random_angles = np.zeros((particle_count, step_count))

    def _generate_random_angles(self):
        """
        Generate random angles for all particles and steps.

        Angles are uniformly distributed between 0 and 2*pi.
        """
        self.random_angles = np.random.uniform(0, 2 * np.pi, size=(self.particle_count, self.step_count))

    def simulate_walks(self):
        """
        Perform the random walk simulation for all particles.
        
        Updates the x and y position arrays with the trajectories of the particles.
        """
        self._generate_random_angles()

        for step in range(self.step_count):
            self.positions_x[:, step + 1] = self.positions_x[:, step] + self.step_size * np.cos(self.random_angles[:, step])
            self.positions_y[:, step + 1] = self.positions_y[:, step] + self.step_size * np.sin(self.random_angles[:, step])

    def calculate_mean_squared_displacement(self):
        """
        Compute the mean squared displacement (MSD) for each step.

        Returns:
        --------
        tuple of np.ndarray
            Mean squared displacement (MSD) and root mean squared displacement (RMSD) for each step.
        """
        displacement_x = self.positions_x - self.positions_x[:, 0:1]
        displacement_y = self.positions_y - self.positions_y[:, 0:1]
        squared_displacement = displacement_x**2 + displacement_y**2

        msd = np.mean(squared_displacement, axis=0)
        rmsd = np.sqrt(msd)

        return msd, rmsd

    def fit_to_power_law(self, msd):
        """
        Fit the mean squared displacement (MSD) data to a power law: MSD = C * n^alpha.

        Parameters:
        -----------
        msd : np.ndarray
            Mean squared displacement data.

        Returns:
        --------
        tuple of float
            Fitted power law exponent (alpha) and coefficient (C).
        """
        step_numbers = np.arange(1, len(msd))
        log_msd = np.log(msd[1:])
        log_steps = np.log(step_numbers)

        coefficients = np.polyfit(log_steps, log_msd, 1)
        alpha = coefficients[0]
        C = np.exp(coefficients[1])

        return alpha, C

    def plot_position_histograms(self, steps_to_plot, bin_sizes):
        """
        Generate 2D histograms of particle positions at specified steps.

        Parameters:
        -----------
        steps_to_plot : list of int
            List of step indices to visualize.
        bin_sizes : list of int
            Number of bins to use for the histograms.
        """
        rows = len(bin_sizes)
        cols = len(steps_to_plot)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        x_min, x_max = np.min(self.positions_x), np.max(self.positions_x)
        y_min, y_max = np.min(self.positions_y), np.max(self.positions_y)

        for i, bins in enumerate(bin_sizes):
            for j, step in enumerate(steps_to_plot):
                ax = axes[i, j]
                histogram = ax.hist2d(
                    self.positions_x[:, step],
                    self.positions_y[:, step],
                    bins=bins,
                    range=[[x_min, x_max], [y_min, y_max]]
                )
                plt.colorbar(histogram[3], ax=ax)
                ax.set_title(f"Step={step}, Bins={bins}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

        plt.suptitle("Particle Position Histograms")
        plt.tight_layout()
        plt.show()

    def plot_cross_section(self, step_index, bins=100):
        """
        Plot x=0 and y=0 cross-sections of particle distributions.

        Parameters:
        -----------
        step_index : int
            Index of the step to visualize.
        bins : int, optional
            Number of bins for the histograms. Default is 100.
        """
        x_positions = self.positions_x[:, step_index]
        y_positions = self.positions_y[:, step_index]

        histogram, x_edges, y_edges = np.histogram2d(x_positions, y_positions, bins=bins)

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        mid_bin_x = bins // 2
        mid_bin_y = bins // 2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(x_centers, histogram[:, mid_bin_y])
        ax1.set_xlabel("x")
        ax1.set_ylabel("Intensity at y=0")
        ax1.grid(True)

        ax2.plot(y_centers, histogram[mid_bin_x, :])
        ax2.set_xlabel("y")
        ax2.set_ylabel("Intensity at x=0")
        ax2.grid(True)

        plt.suptitle(f"Cross-sections at Step {step_index}")
        plt.show()

    def compute_2d_histogram(self, step_number=None, bins=100):
        """
        Calculate a 2D histogram of particle positions at a specific simulation step.

        Parameters:
        -----------
        step_number : int, optional
            The step index to compute the histogram for. Defaults to the final step if not provided.
        bins : int, optional
            The number of bins to use for both x and y directions. Default is 100.

        Returns:
        --------
        tuple of np.ndarray
            Histogram data and the edges of bins along the x and y axes.
        """
        # Default to the final step if no step number is specified
        if step_number is None:
            step_number = self.step_count

        # Extract the particle positions at the specified step
        x_positions = self.positions_x[:, step_number]
        y_positions = self.positions_y[:, step_number]

        # Generate the 2D histogram
        hist, x_edges, y_edges = np.histogram2d(x_positions, y_positions, bins=bins)

        return hist, x_edges, y_edges

if __name__ == "__main__":
    num_steps = 10000
    particle_numbers = [10, 100, 1000, 10000]
    step_length = 1.0

    for particle_count in particle_numbers:
        print(f"Simulating {particle_count} particles...")
        simulator = RandomWalkSimulator(particle_count, num_steps, step_length)
        simulator.simulate_walks()
        msd, rmsd = simulator.calculate_mean_squared_displacement()

        alpha, C = simulator.fit_to_power_law(msd)
        print(f"Particles: {particle_count}, alpha: {alpha:.4f}, C: {C:.4f}")
        

# Test different particle numbers
num_steps = 10000
particle_numbers = [10, 100, 1000, 10000]
step_length = 1.0
results = []

for particle_count in particle_numbers:
    # Initialize the simulator and run the random walk simulation
    print(f"Running simulation with {particle_count} particles...")
    simulator = RandomWalkSimulator(particle_count=particle_count, step_count=num_steps, step_size=step_length)
    simulator.simulate_walks()

    # Calculate the mean squared displacement (MSD) and root mean squared displacement (RMSD)
    msd, rmsd = simulator.calculate_mean_squared_displacement()
    results.append((msd, rmsd))

# Plot the results
steps = np.arange(num_steps + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the Mean Square Displacement (MSD) for different particle numbers
for i, particle_count in enumerate(particle_numbers):
    ax1.plot(steps, results[i][0], label=f'# of particles = {particle_count}')
ax1.set_xlabel('Steps')
ax1.set_ylabel(r'$<r^2>$')
ax1.set_title('Mean Square Displacement (MSD)')
ax1.legend()
ax1.grid(True)

# Plot the Root Mean Square Displacement (RMSD) for different particle numbers
for i, particle_count in enumerate(particle_numbers):
    ax2.plot(steps, results[i][1], label=f'# of particles = {particle_count}')
ax2.set_xlabel('Steps')
ax2.set_ylabel(r'$\sqrt{<r^2>}$')
ax2.set_title('Root Mean Square Displacement (RMSD)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
