from random import random, randrange
from matplotlib.cm import plasma  # For color mapping

import numpy as np
import matplotlib.pyplot as plt

class QuantumSimulator:
    """
    Simulation of N independent particles in a 1D infinite potential well using Monte Carlo methods.

    Attributes:
        particle_count (int): Total number of particles in the system.
        ground_state_energy (float): Energy of the ground state per particle.
        kbT (float): Dimensionless kbT.
        steps (int): Total number of Monte Carlo iterations.
        beta (float): Inverse kbT (1/kbT).
        quantum_states (numpy.ndarray): Array of quantum states for each particle.
        energy_history (numpy.ndarray): Array to track total energy over time.
    """

    def __init__(self, particle_count=1000, ground_state_energy=1.0, kbT=10.0, steps=10000):
        """
        Initialize the quantum system and simulation parameters.

        Args:
            particle_count (int, optional): Number of particles. Defaults to 1000.
            ground_state_energy (float, optional): Ground state energy. Defaults to 1.0.
            kbT (float, optional): System kbT. Defaults to 10.0.
            steps (int, optional): Number of Monte Carlo iterations. Defaults to 10000.
        """
        self.particle_count = particle_count  # Total number of particles
        self.ground_state_energy = ground_state_energy  # Energy of the ground state per particle
        self.kbT = kbT  # Dimensionless kbT of the system
        self.steps = steps  # Total simulation steps
        self.beta = 1.0 / kbT  # Inverse kbT

        # Initialize particle quantum states and energy tracking
        self.quantum_states = np.ones(particle_count, dtype=int)  # Start all particles in ground state
        self.energy_history = np.zeros(steps)  # Track energy at each step
        self.energy_history[0] = self.calculate_total_energy()  # Initial system energy

    def calculate_energy_difference(self, state, change):
        """
        Compute the change in energy from a quantum state transition.

        Args:
            state (int): Current quantum state.
            change (int): Direction of transition (+1 for up, -1 for down).

        Returns:
            float: Energy difference due to the transition.
        """
        new_state = state + change
        initial_energy = self.ground_state_energy * state**2
        final_energy = self.ground_state_energy * new_state**2
        return final_energy - initial_energy

    def calculate_total_energy(self):
        """
        Compute the total energy of the system.

        Returns:
            float: Total energy of all particles.
        """
        return self.ground_state_energy * np.sum(self.quantum_states**2)

    def attempt_transition(self, particle_index, current_step):
        """
        Attempt a quantum state transition for a given particle.

        Args:
            particle_index (int): Index of the particle to attempt a transition.
            current_step (int): Current Monte Carlo iteration index.
        """
        direction = 1 if random() < 0.5 else -1  # Determine transition direction

        # Prevent downward transition from the ground state
        if self.quantum_states[particle_index] == 1 and direction == -1:
            self.energy_history[current_step] = self.energy_history[current_step - 1]
            return

        # Calculate energy difference for the proposed transition
        delta_energy = self.calculate_energy_difference(self.quantum_states[particle_index], direction)

        if direction == -1:  # Downward transition
            if self.quantum_states[particle_index] > 1:  # Allowed if not in ground state
                self.quantum_states[particle_index] += direction
                self.energy_history[current_step] = self.energy_history[current_step - 1] + delta_energy
            else:
                self.energy_history[current_step] = self.energy_history[current_step - 1]
        else:  # Upward transition
            acceptance_probability = np.exp(-self.beta * delta_energy)
            if random() < acceptance_probability:  # Accept based on probability
                self.quantum_states[particle_index] += direction
                self.energy_history[current_step] = self.energy_history[current_step - 1] + delta_energy
            else:
                self.energy_history[current_step] = self.energy_history[current_step - 1]

    def execute_simulation(self):
        """
        Perform the Monte Carlo simulation.

        Returns:
            numpy.ndarray: Energy history of the system over time.
        """
        for step in range(1, self.steps):
            particle = randrange(self.particle_count)  # Select a random particle
            self.attempt_transition(particle, step)  # Attempt a quantum state change

        return self.energy_history

# Run the simulation
simulator = QuantumSimulator()
energy_trajectory = simulator.execute_simulation()
print(f"Final system energy: {energy_trajectory[-1]}")

# Plot results of the simulation
plt.figure(figsize=(10, 6))
plt.plot(range(simulator.steps), energy_trajectory)
plt.xlabel('Monte Carlo Iteration')
plt.ylabel('Total System Energy')
plt.title('Energy Evolution in Quantum System')
plt.grid(True)
plt.show()

# Perform multiple runs and analyze results
plt.figure(figsize=(10, 6))
num_simulations = 20  # Number of independent runs
colors = plasma(np.linspace(0, 1, num_simulations))  # Assign colors for plots
final_energies = []  # Collect final energies of each run

# Simulation parameters
particle_count = 1000
ground_state_energy = 1.0
kbT = 100.0  # High kbT regime
steps = 100000  # Longer simulation for convergence

for run in range(num_simulations):
    simulation = QuantumSimulator(particle_count, ground_state_energy, kbT, steps)
    trajectory = simulation.execute_simulation()
    final_energies.append(trajectory[-1])
    plt.plot(range(simulation.steps), trajectory, color=colors[run], alpha=0.7, label=f'Run {run + 1}')

# Calculate statistics
mean_energy = np.mean(final_energies)
std_dev_energy = np.std(final_energies)
print(f"Average final energy: {mean_energy}")
print(f"Standard deviation: {std_dev_energy}")

mean_energy_per_particle = mean_energy / particle_count
print(f"Mean energy per particle: {mean_energy_per_particle}")
print(f"Expected energy per particle: {kbT / 2}")

plt.xlabel('Monte Carlo Iteration')
plt.ylabel('Total System Energy')
plt.title('Multiple Runs of Quantum System Simulation')
plt.grid(True)
plt.show()