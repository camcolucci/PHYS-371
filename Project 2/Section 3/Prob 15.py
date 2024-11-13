import numpy as np
from FourthOrderRungeKutta import RK4
import matplotlib.pyplot as plt

# Define the constants
hbar = 1  # Reduced Planck's constant (in atomic units)
mu = 1    # Reduced mass of the diatomic molecule (in atomic units)
x_0 = 1   # Equilibrium bond distance (in atomic units)
D = 100 * (hbar**2 / (2 * mu * x_0**2))  # Depth of the potential well
a = 0.7 / x_0  # Morse parameter, controls the width of the potential well

# Define the Morse potential function V(x)
def morse_function(x, D, a, x_0):
    """
    Computes the Morse potential V(x) at a given position x.
    """
    arg1 = np.clip(-2 * a * (x - x_0), -100, 100)
    arg2 = np.clip(-a * (x - x_0), -100, 100)
    
    return D * (np.exp(arg1) - 2 * np.exp(arg2))

# Define the ODEs and system of equations for the Schrödinger equation
def schrodinger(state, x, E):
    """
    Defines the system of differential equations for the Schrödinger equation.
    """
    psi, phi = state  # Unpack the state vector
    V = morse_function(x, D, a, x_0)
    dpsi_dx = phi  # First derivative of the wavefunction (dpsi/dx)
    dphi_dx = (2 * mu / hbar**2) * (V - E) * psi  # Second derivative from the Schrödinger equation
    return np.array([dpsi_dx, dphi_dx])

# Function to solve for psi with a given energy E and return the wavefunction psi
def solve_for_psi(E):
    """
    Solves the Schrödinger equation for a given energy E and returns the x and psi values.
    """
    # Set initial conditions for the wavefunction and its derivative
    a = 0       # Start at x = 0 (the actual value is the diameter of the particle, so small we assume 0)
    b = 3.0     # End point for x, defining the observation range
    N = 300    # Number of steps
    psi_0 = 1.0 # Initial value for wavefunction (psi)
    phi_0 = 0.0 # Initial slope of wavefunction (dpsi/dx)
    
    # Use the RK4 solver with the Schrödinger equation
    rk4_solver = RK4(func=lambda state, x: schrodinger(state, x, E), a=a, b=b, N=N, x_0=[psi_0, phi_0])
    
    # Solve the system using RK4
    x_values, solution = rk4_solver.solve()
    if len(x_values) < N:  # Checks the solver to see if the function diverged
        print(f"Solver diverged for E = {E:.4f}")
        return np.array([]), np.array([])  # Return empty arrays if diverged
    return x_values, solution[:, 0]

# Initial parameters for finding the lowest eight eigenvalues states
eigenvalues = []      # List to store the lowest eight eigenvalues
eigenfunctions = []   # List to store the corresponding wavefunctions
num_bound_states = 8  # We need the lowest eight bound states
tolerance = 1e-5      # Tolerance for determining if psi decays to zero
E = -D / 2            # Initial guess for energy, starting halfway down the potential well
step_size = 0.05      # Step size for incrementing the energy guess

# Loop to find the lowest eight eigenvalues
while len(eigenvalues) < num_bound_states:
    x_values, psi_values = solve_for_psi(E)
    
    if x_values.size == 0:  # Diverged, skip to next E
        E += step_size
        continue
    
    psi_end = psi_values[-1]  # Check the endpoint value of psi to see if it decays to zero
    
    # Check if psi decays near zero at the endpoint
    if abs(psi_end) < tolerance:
        # If psi approaches zero, the eigenvalue and eigenfunction is stored
        eigenvalues.append(E)
        eigenfunctions.append(psi_values)
        print(f"Found Eigenstate {len(eigenvalues)} with E = {E:.4f}")
        # Increase energy guess to find the next bound state
        E += 0.5
    else:
        # Adjust energy by small increments to search for the next bound state
        # Reduce step size when getting close to an eigenvalue
        if abs(psi_end) < 10 * tolerance:
            E += step_size / 2  # Make the step smaller when close
        else:
            E += step_size

# Print the eigenvalues for the eight lowest eigenstates
for i, energy in enumerate(eigenvalues, start=1):
    print(f"Eigenvalue {i}: E = {energy:.4f}")


plt.figure(figsize=(10, 6))

# Compute Morse potential values over the range of x values
morse_potential_values = morse_function(x_values, D, a, x_0)

# Plot the Morse potential
plt.plot(x_values, morse_potential_values, label="Morse Potential $V(x)$", color='black', linestyle='-', linewidth=1.5)

# Plot the wavefunctions for the eigenstates
for i, psi_values in enumerate(eigenfunctions):
    # Normalize the wavefunctions for visualization
    norm_psi_values = psi_values / max(abs(psi_values)) * 0.5 * D
    plt.plot(x_values, norm_psi_values, label=f"$\\psi(x)$ for eigenvalue {i+1}: E = {eigenvalues[i]:.4f}")

plt.xlabel("Displacement (x)")
plt.ylabel("Wavefunction $\\psi(x)$")
plt.title("Wavefunctions and Morse Potential Overlay")
plt.legend()
plt.grid(True)
plt.show()
