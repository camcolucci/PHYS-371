import numpy as np

"""Function that takes inputs omega_0, omega_d and f_0 as parameters to return 
the right side of the damped oscillator equation
"""
def linear_oscillator(omega_0, omega_d, f_0):
    dxdt = v                                                            # Defines the first derivative as velocity
    dvdt = -2 * beta * v - omega_0**2 * x + f_0 * np.sin(omega_d * t)   # Define the equation of the damped oscillator
    return dxdt, dvdt                                                   # Returns values for the velocity and the 
                                                                            # acceleration of the oscillator