import numpy as np
import matplotlib.pyplot as plt

# Initialize the variables that will define the function.

a = 0                          # Starting point
b = 30                         # Ending point
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
        n -= 1
    return n

# Define Green's function G(t-t')

def G(t, t_prime, beta):
    tau = t - t_prime
    if tau < 0:
        return 0
    return (np.exp(-beta * tau) * np.sin(omega_0 * tau))

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
    dt_prime = t_prime_value[1] - t_prime_value[0]                                               # Set local step size
    calculated_value = [F(t_prime, alpha) * G(t, t_prime, beta) for t_prime in t_prime_value]    # Function and Greens funtion values
    return simpsons_rule(calculated_value, dt_prime)                                                  

# Compute x(t) while incorporating the different beta and alpha values

def vary_beta_and_alpha(print_values=False):    #beta = 0.1 * omega_0, alpha = 0.3 * omega_0
    beta_1 = 0.1 * omega_0
    alpha_1 = 0.3 * omega_0
    x_t_1 = np.array([compute_SHO(ti, beta_1, alpha_1) for ti in t])

                                                #beta = 0.2 * omega_0, alpha = 0.2 * omega_0
    beta_2 = 0.2 * omega_0
    alpha_2 = 0.2 * omega_0
    x_t_2 = np.array([compute_SHO(ti, beta_2, alpha_2) for ti in t])

                                                #beta = 0.3 * omega_0, alpha = 0.1 * omega_0
    beta_3 = 0.3 * omega_0
    alpha_3 = 0.1 * omega_0
    x_t_3 = np.array([compute_SHO(ti, beta_3, alpha_3) for ti in t])

# Lets the functions print if requested
    if print_values:
        print(f"x(t) for beta={beta_1:.2f}, alpha={alpha_1:.2f}:")
        print(x_t_1)
        print(f"x(t) for beta={beta_2:.2f}, alpha={alpha_2:.2f}:")
        print(x_t_2)
        print(f"x(t) for beta={beta_3:.2f}, alpha={alpha_3:.2f}:")
        print(x_t_3)

    return x_t_1, x_t_2, x_t_3

# Call the function to compute the values and print them
x_t_1, x_t_2, x_t_3 = vary_beta_and_alpha(print_values=True)

# Plot the results as an overlayed graph containing all three outputs

plt.plot(t, x_t_1, label=r'$\beta={:.2f}, \alpha={:.2f}$'.format(0.1 * omega_0, 0.3 * omega_0))
plt.plot(t, x_t_2, label=r'$\beta={:.2f}, \alpha={:.2f}$'.format(0.2 * omega_0, 0.2 * omega_0))
plt.plot(t, x_t_3, label=r'$\beta={:.2f}, \alpha={:.2f}$'.format(0.3 * omega_0, 0.1 * omega_0))
plt.xlabel("Time (t)")
plt.ylabel("Displacement of the mass x(t)")
plt.title("Potential Energy Function (Harmonic Oscillator)")
plt.grid(True)
plt.legend(loc="best")
plt.show()

#You'll notice that as beta increases and alpha decreases the system rapidly dampens back to 0