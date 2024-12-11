# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name
from random import random, randrange
from matplotlib.cm import viridis  # or 'plasma', 'magma', 'inferno'

import numpy as np
import matplotlib.pyplot as plt



class MonteCarloGas:
    """
    Monte Carlo simulation of N non-interacting point particles trapped in a 
    one-dimensional infinite well of length L.
    
    ## Keyword arguments
    --------------------
    N : |int|
        Total number of particles
    epsilon1 : |float|
        Ground state energy (per particle)
    kT : |float|
        Temperature in dimensionless energy units
    ns : |int|
        Total number of Monte Carlo sweeps
    """
    
    def __init__(self, N=1000, epsilon1=1.0, kT=10.0, ns=10000):
        self.N = N # Total number of particles
        self.epsilon1 = epsilon1 # Ground state energy
        self.kT = kT # Temperature (kB*T)
        self.ns = ns # Number of Monte Carlo sweeps
        self.beta = 1.0/kT # Inverse temperature
        
        # Initialize arrays
        self.n = np.ones(N, dtype=int) # Quantum numbers (all particles start in ground state)
        self.E = np.zeros(ns) # Total energy at each step
        self.E[0] = self.compute_total_energy()  # Store initial energy

        
    def compute_energy_change(self, n, direction):
        """
        Compute energy change for transition n -> n +- 1.
        
        ## Keyword arguments
        --------------------
        n : |int|
            Current quantum number
        direction : |int|
            +1 for increase, -1 for decrease
        
        ## Returns
        ----------
        |float| : Energy change dE
        """
        n_final = n+direction
        E_initial = self.epsilon1*n**2
        E_final = self.epsilon1*n_final**2
        return E_final-E_initial
        
    def compute_total_energy(self):
        """
        Compute total energy for current state.
        
        ## Returns
        ----------
        |float| : Total energy
        """
        return self.epsilon1*np.sum(self.n**2)
    
    def try_transition(self, i, step):
        """
        Attempt a transition for particle i at Monte Carlo step.
        
        ## Keyword arguments
        --------------------------
        i : |int|
            Particle index
        step : |int|
            Current Monte Carlo step
        """
        # Randomly choose up or down transition
        direction = 1 if random() < 0.5 else -1
        
        # Handle ground state case
        if self.n[i] == 1 and direction == -1: # Is in ground state and trying to go down
            self.E[step] = self.E[step-1] # Energy does not change
            return
            
        # Compute energy change
        dE = self.compute_energy_change(self.n[i], direction)
        
        # Handle transitions:
        # Downward transitions are always accepted if not in ground state
        #   as they are energetically favorable.

        if direction == -1: # Downward transition
            if self.n[i] > 1: # Accept with certainty if not in ground state
                self.n[i] += direction # Update quantum number
                self.E[step] = self.E[step-1]+dE # Update total energy
            else: # Reject transition
                self.E[step] = self.E[step-1] # Energy does not change
                
        # Upward transitions are accepted with Metropolis probability
        #   P_accept = exp(-beta*dE)
        #   where beta = 1/kT
        
        else: # Upward transition
            # Metropolis acceptance probability
            P_accept = np.exp(-self.beta*dE)
            if random() < P_accept: # Accept transition if random number (0-1) is less than P_accept
                self.n[i] += direction
                self.E[step] = self.E[step-1]+dE
            else: # Reject transition
                self.E[step] = self.E[step-1]
                
    def run_simulation(self):
        """
        Run the full Monte Carlo simulation.
        
        ## Returns
        ----------
        |np.ndarray| : Array of energy values at each step
        """
        for step in range(1, self.ns):
            # Randomly select particle
            i = randrange(self.N)
            # Try transition
            self.try_transition(i, step)
            
        return self.E
    
# Create instance and run simulation
mc_gas = MonteCarloGas()
energies = mc_gas.run_simulation()
print(f"Final energy: {energies[-1]}")

# Plot results for problem 13
plt.figure(figsize=(10, 6))
plt.plot(range(mc_gas.ns), energies)
plt.xlabel('Monte Carlo Step')
plt.ylabel('Total Energy')
plt.title('Energy Evolution in 1D Quantum Gas')
plt.grid(True)
plt.show()

# Run multiple simulations and plot for problem 14-16
plt.figure(figsize=(10, 6))
num_runs = 10 # Number of runs of the simulation
colors = viridis(np.linspace(0, 1, num_runs)) # Generate colors for each run
final_energies = [] # Store final energies for each run for comparison

# Parameters for the simulation to solve the <E> =/= kT/2 problem
N = 1000
epsilon1 = 1.0
kT = 100.0 # kT >> epsilon1 for high temperature limit
ns = 500000 # Large number of steps for convergence

for i in range(num_runs):  # Run num_runs number of simulations
    mc_gas = MonteCarloGas(N, epsilon1, kT, ns) # Remove input parameters to use defaults
    energies = mc_gas.run_simulation()
    # print(f"Final energy for run {i+1}: {energies[-1]}")
    final_energies.append(energies[-1])
    plt.plot(range(mc_gas.ns), energies, 
             color=colors[i], alpha=0.7,
             label=f'Run {i+1}')

# Calculate mean and standard deviation
mean = np.mean(final_energies)
std = np.std(final_energies)

print(f"Mean final energy: {mean}")
print(f"Standard deviation: {std}")

# The energy per particle is mean/N
mean_per_particle = mean/N
print(f"Mean final energy per particle: {mean_per_particle}")
print(f"Expected energy per particle: {kT/2}")

# The standard deviation per particle is std/N
std_per_particle = std/N
print(f"Standard deviation per particle: {std_per_particle}")

plt.xlabel('Monte Carlo Step')
plt.ylabel('Total Energy')
plt.title('Multiple Runs of 1D Quantum Gas Simulation')
plt.grid(True)
plt.show()