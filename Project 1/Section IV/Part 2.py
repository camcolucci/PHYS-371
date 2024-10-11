
import numpy as np

"""Function that takes inputs omega_0, omega_d and f_0 as parameters to return 
the right side of the damped oscillator equation
"""
beta = gamma * omega_0

def linear_oscillator(x, v, omega_0, omega_d, f_0, t):
    
    dxdt = v   # Defines the first derivative as velocity
    
    dvdt = -2 * beta * v - omega_0**2 * x + f_0 * np.sin(omega_d * t)  # Define the equation of the damped oscillator
                                                                
    return dxdt, dvdt
                                                                            
                                                                          