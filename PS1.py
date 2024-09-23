import numpy as np
import matplotlib.pyplot as plt

def generate_points(N, R):
    # Generate N random points within the square
    x = np.random.uniform(-R, R, N)
    y = np.random.uniform(-R, R, N)
    return x, y

def estimate_pi(N, R):
    x, y = generate_points(N, R)  # Generate random points
    inside_circle = x**2 + y**2 < R**2  # Check if points are inside the circle
    N_c = np.sum(inside_circle)  # Count the number of points inside the circle
    pi_estimate = 4 * N_c / N  # Estimate the value of π
    return pi_estimate

# Define a range of N values
N_values = [100, 1000, 5000, 10000, 50000, 100000, 500000]
R = 1
pi_estimates = []
differences = []

# Calculate π estimate and differences for each N value
for N in N_values:
    pi_estimate = estimate_pi(N, R)
    pi_estimates.append(pi_estimate)
    difference = pi_estimate - np.pi  # Difference between estimated and actual π
    differences.append(difference)
    print(f"N = {N}, Estimated π: {pi_estimate}, Difference: {difference}")

# Plotting the differences as a function of N
plt.figure(figsize=(10, 6))
plt.plot(N_values, differences, marker='o', linestyle='-', label='Difference (Estimated π - Actual π)')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Difference Line')
plt.xscale('log')  # Logarithmic scale for better visualization
plt.xlabel('Number of Points (N)')
plt.ylabel('Difference from Actual π')
plt.title('Difference between Estimated and Actual π as N Increases')
plt.legend()
plt.grid(True)
plt.show()


