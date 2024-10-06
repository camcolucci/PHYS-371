import numpy as np
import matplotlib.pyplot as plt

# Initialize the variables that will define the function.

a = 0                          # Starting point
b = 20                         # Ending point
n = 1000                       # Number of iterations, must be even
m = 1                          # Mass of the block in the oscillator
k = 1                          # Spring constant
B = 1                          # Damping Factor
beta = B / (2*m)               # Damping Coefficient
omega_0 = np.sqrt(k / m)       # Natural Frequency
t = np.linspace(a, b, n)       # Creates time values for the function
alpha = 0.1                    # Decay constant of the driving force
dt = t[1] - t[0]               # Change in time/step size

# Define the force function F(t')

def F(t_prime, alpha):                
    return m * np.exp(-alpha * t_prime)
  
# Ensures that n is even for Simpson's rule
  
def subintervals(n):          
    if n % 2 == 1:
        n -= 1          #If n is an odd number it removes 1 to then make it an even number of iterations
    return n

# Define Green's function G(t-t')

def G(t, t_prime, beta):
    tau = t - t_prime                                                               #Change in time
    if tau < 0:
        return 0
    return (1 / (m * omega_0)) * np.exp(-beta * tau) * np.sin(omega_0 * tau)        #Using the Green's function definition

# Set up Simpson's rule formula

def simpsons_rule(functions, h):
    n = len(functions)
    result = functions[0] + functions[-1]       # First and last terms
    result += 4 * np.sum(functions[1:n:2])      # Odd terms
    result += 2 * np.sum(functions[2:n-1:2])    # Even terms
    return h/3 * result

# Find the total displacement x(t)

def compute_SHO(t, beta, alpha):
    t_prime_value = np.linspace(a, t, n)                                                         # Time points for the observed system
    calculated_value = [F(t_prime, alpha) * G(t, t_prime, beta) for t_prime in t_prime_value]    # Function and Greens funtion values
    return simpsons_rule(calculated_value, dt)                                                   

#Find the velocity of the function

def velocity(x, dt):
    return np.gradient(x, dt)

# Calculate the total energy of the function

def energy(x, v, m, k):
    kinetic = 0.5 * m * v**2                    #Kinetic Energy
    potential = 0.5 * k * x**2                  #Potential Energy in the spring
    total_energy = kinetic + potential          #Total energy of the system
    return total_energy

# Compute x(t) while incorporating the different beta and alpha values

def vary_beta_and_alpha():                     #beta = 0.1 * omega_0, alpha = 0.3 * omega_0
    beta_1 = 0.1 * omega_0
    alpha_1 = 0.3 * omega_0
    x_t_1 = np.array([compute_SHO(ti, beta_1, alpha_1) for ti in t])
    v_t_1 = velocity(x_t_1, dt)
    E_t_1 = energy(x_t_1, v_t_1, m, k)

                                                #beta = 0.2 * omega_0, alpha = 0.2 * omega_0
    beta_2 = 0.2 * omega_0
    alpha_2 = 0.2 * omega_0
    x_t_2 = np.array([compute_SHO(ti, beta_2, alpha_2) for ti in t])
    v_t_2 = velocity(x_t_2, dt)
    E_t_2 = energy(x_t_2, v_t_2, m, k)

                                                #beta = 0.3 * omega_0, alpha = 0.1 * omega_0
    beta_3 = 0.3 * omega_0
    alpha_3 = 0.1 * omega_0
    x_t_3 = np.array([compute_SHO(ti, beta_3, alpha_3) for ti in t])
    v_t_3 = velocity(x_t_3, dt)
    E_t_3 = energy(x_t_3, v_t_3, m, k)

    return E_t_1, E_t_2, E_t_3

# Call the function to compute the values and print them
E_t_1, E_t_2, E_t_3 = vary_beta_and_alpha()

# Plot the results as an overlayed graph containing all three outputs

plt.plot(t, E_t_1, label=r'$\beta={:.2f}, \alpha={:.2f}$'.format(0.1 * omega_0, 0.3 * omega_0))
plt.plot(t, E_t_2, label=r'$\beta={:.2f}, \alpha={:.2f}$'.format(0.2 * omega_0, 0.2 * omega_0))
plt.plot(t, E_t_3, label=r'$\beta={:.2f}, \alpha={:.2f}$'.format(0.3 * omega_0, 0.1 * omega_0))
plt.xlabel("Time (t)")
plt.ylabel("Total Energy $E(t)$")
plt.title("Total Energy in the Damped Harmonic Oscillator")
plt.grid(True)
plt.legend(loc="best")
plt.show()