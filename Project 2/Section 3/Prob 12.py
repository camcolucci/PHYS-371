import numpy as np
from FourthOrderRungeKutta import RK4

# Define the constants
hbar = 1  # Reduced Planck's constant (in atomic units, so we set hbar = 1)
mu = 1    # Reduced mass of the diatomic molecule (in atomic units, so mu = 1)
x_0 = 1   # Equilibrium bond distance (in atomic units, so x_0 = 1)
D = 100 * (hbar**2 / (2 * mu * x_0**2))  # Depth of the potential well
a = 0.7 / x_0  # Morse parameter, controlling the width of the potential well

# Define the Morse potential function V(x)
def morse_function(x, D, a, x_0):
    """
    Computes the Morse potential V(x) at a given position x.
    
    Parameters:
    x : float
        Position at which to calculate the Morse potential.
    D : float
        Depth of the potential well.
    a : float
        Width parameter of the Morse potential.
    x_0 : float
        Equilibrium bond distance (center of the potential well).
    
    Returns:
    float
        The value of the Morse potential at position x.
    """
    return D * (np.exp(-2 * a * (x - x_0)) - 2 * np.exp(-a * (x - x_0)))

# Define the ODEs and system of equations for the Schrödinger equation
def schrodinger(state, x, E):
    """
    Defines the system of differential equations for the Schrödinger equation.
    
    Parameters:
    state : array-like
        The current state vector [psi, phi] where psi is the wavefunction
        and phi is its derivative dpsi/dx.
    x : float
        The current position.
    E : float
        The energy eigenvalue guess for the wavefunction.

    Returns:
    array
        Array containing the derivatives [dpsi/dx, dphi/dx].
    """
    psi, phi = state  # Unpacks the state vector
    V = morse_function(x, D, a, x_0)
    dpsi_dx = phi  # First derivative of the wavefunction (dpsi/dx)
    dphi_dx = (2 * mu / hbar**2) * (V - E) * psi  # Second derivative from the Schrödinger equation
    return np.array([dpsi_dx, dphi_dx])

# Set initial conditions for the wavefunction and its derivative
a = 0       # Start at x = 0 (the actual value is the diameter of the particle which is so small we can assume 0)
b = 5.0     # End point for x, defining the observation range
N = 1000    # Number of steps
psi_0 = 1.0 # Initial value for wavefunction (psi)
phi_0 = 0.0 # Initial slope of wavefunction (dpsi/dx)

# Initial guess for energy E using the half max value of the potential well
E = -D / 2 

# Initialize the RK4 solver with the Schrödinger equation
# Use a lambda function to pass the energy value E to the schrodinger function
rk4_solver = RK4(func=lambda state, x: schrodinger(state, x, E), 
                 a=a, b=b, N=N, x_0=[psi_0, phi_0])

# Solve the system using RK4
x_values, solution = rk4_solver.solve()

# Get psi values (wavefunction) from the solution
psi_values = solution[:, 0]

# Output psi values, we should see them oscillate about 0
for x, psi in zip(x_values, psi_values):
    print(f"x: {x:.4e} m, psi(x): {psi:.4e}")
