import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from RandomWalk import RandomWalkSimulator

class DiffusionAnalysis2D:
    """
    DiffusionAnalysis2D processes diffusion data to extract power spectra 
    and autocorrelation functions from simulated random walks.
    """
    def __init__(self, histogram=None, x_bins=None, y_bins=None):
        """
        Initialize the analysis with histogram data.

        Keyword Arguments:
        ------------------
        histogram : np.ndarray, optional
            2D histogram data for intensity I(x, y).
        x_bins : np.ndarray, optional
            Edges of the bins along the x-axis.
        y_bins : np.ndarray, optional
            Edges of the bins along the y-axis.
        """
        self.histogram = histogram
        self.x_bins = x_bins
        self.y_bins = y_bins

    @staticmethod
    def simulate_and_save(file_name, num_particles, num_steps, step_size, bin_count):
        """
        Run the random walk simulation and save the histogram data as a pickle file.

        Keyword Arguments:
        ------------------
        file_name : str
            Path to save the pickle file.
        num_particles : int
            Number of particles.
        num_steps : int
            Number of steps per particle.
        step_size : float
            Step size.
        bin_count : int
            Number of bins for the histogram.
        """
        Rand_Wlk_simulation = RandomWalkSimulator(particle_count=num_particles, step_count=num_steps, step_size=step_size)
        Rand_Wlk_simulation.simulate_walks()
        histogram, x_bins, y_bins = Rand_Wlk_simulation.compute_2d_histogram(bins=bin_count)

        data = {'hist': histogram, 'x_edges': x_bins, 'y_edges': y_bins}
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load_data(file_name):
        """
        Load pickled histogram data from file.

        Keyword Arguments:
        ------------------
        file_name : str
            Path to the pickle file.

        Returns:
        --------
        tuple : (histogram, x_bins, y_bins)
        """
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        return data['hist'], data['x_edges'], data['y_edges']

    def plot_slices(self):
        """
        Plot slices of I(x, y) along x=0 and y=0.
        """
        x_midpoints = (self.x_bins[:-1] + self.x_bins[1:]) / 2
        y_midpoints = (self.y_bins[:-1] + self.y_bins[1:]) / 2

        x_profile = self.histogram[self.histogram.shape[0] // 2, :]
        y_profile = self.histogram[:, self.histogram.shape[1] // 2]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_midpoints, x_profile)
        plt.title("I(x=0, y) Slice")
        plt.xlabel("y")
        plt.ylabel("Intensity")

        plt.subplot(1, 2, 2)
        plt.plot(y_midpoints, y_profile)
        plt.title("I(x, y=0) Slice")
        plt.xlabel("x")
        plt.ylabel("Intensity")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_power_spectrum(profile_data):
        """
        Compute the 1D power spectrum of a slice using FFT.

        Keyword Arguments:
        ------------------
        profile_data : np.ndarray
            1D data slice (e.g., I(x=0, y) or I(x, y=0)).

        Returns:
        --------
        tuple : (wave_numbers, spectrum)
            Wave numbers and power spectrum values.
        """
        fft_output = np.fft.fft(profile_data)
        spectrum = np.abs(fft_output)**2

        wave_numbers = np.fft.fftshift(np.fft.fftfreq(len(profile_data)))
        spectrum = np.fft.fftshift(spectrum)

        return wave_numbers, spectrum

    def compute_1d_autocorrelation(self, spectrum):
        """
        Compute 1D autocorrelation function from power spectrum using IFFT.

        Keyword Arguments:
        ------------------
        spectrum : np.ndarray
            1D power spectrum data.

        Returns:
        --------
        np.ndarray : Autocorrelation function values.
        """
        autocorr_function = np.fft.ifft(np.fft.ifftshift(spectrum)).real
        return autocorr_function

    def compute_2d_autocorrelation(self):
        """
        Compute the 2D autocorrelation function using FFT.

        Returns:
        --------
        np.ndarray : 2D autocorrelation function values.
        """
        fft_2d = np.fft.fft2(self.histogram)
        spectrum_2d = np.abs(fft_2d)**2
        autocorr_2d = np.fft.fftshift(np.fft.ifft2(spectrum_2d).real)
        autocorr_2d /= np.max(autocorr_2d)

        return autocorr_2d

    def plot_2d_autocorrelation(self, autocorr_2d):
        """
        Plot the 2D autocorrelation function as a contour plot.

        Keyword Arguments:
        ------------------
        autocorr_2d : np.ndarray
            2D autocorrelation function values.
        """
        plt.figure(figsize=(8, 6))
        plt.contourf(autocorr_2d, cmap='viridis')
        plt.colorbar(label="Autocorrelation")
        plt.title("2D Autocorrelation Function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    @staticmethod
    def extract_1d_slices(histogram, x_bins, y_bins):
        """
        Extract 1D slices from the 2D histogram data along x=0 and y=0.

        Keyword Arguments:
        ------------------
        histogram : np.ndarray
            2D histogram data.
        x_bins : np.ndarray
            Bin edges along the x-axis.
        y_bins : np.ndarray
            Bin edges along the y-axis.

        Returns:
        --------
        tuple : (x_profile, y_profile, x_midpoints, y_midpoints)
            1D slices along x=0 and y=0, and the corresponding bin centers.
        """
        x_midpoints = (x_bins[:-1] + x_bins[1:]) / 2
        y_midpoints = (y_bins[:-1] + y_bins[1:]) / 2

        x_zero_idx = np.abs(x_midpoints).argmin()
        y_zero_idx = np.abs(y_midpoints).argmin()

        x_profile = histogram[x_zero_idx, :]
        y_profile = histogram[:, y_zero_idx]

        return x_profile, y_profile, x_midpoints, y_midpoints

# Parameters
file_name = "diffusion_data.pkl"
num_particles = 10000
num_steps = 10000
step_size = 1.0
bin_count = 100
current_step = num_steps

if os.path.exists(file_name):
    print(f"The results file '{file_name}' already exists.")
    recalculate = input("Recalculate the random walk data? [y,n]: ").strip().lower()
    if recalculate == 'y':
        DiffusionAnalysis2D.simulate_and_save(file_name, num_particles, num_steps, step_size, bin_count)
else:
    DiffusionAnalysis2D.simulate_and_save(file_name, num_particles, num_steps, step_size, bin_count)

histogram, x_bins, y_bins = DiffusionAnalysis2D.load_data(file_name)

plt.imshow(histogram.T, origin='lower', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], aspect='auto')
plt.colorbar(label="Intensity")
plt.title(f"Histogram of 2D Particle Distribution after {current_step} Steps")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x_profile, y_profile, x_midpoints, y_midpoints = DiffusionAnalysis2D.extract_1d_slices(histogram, x_bins, y_bins)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_midpoints, y_profile)
plt.title("Intensity Slice of I(x=0, y)")
plt.xlabel("y")
plt.ylabel("Intensity")

plt.subplot(1, 2, 2)
plt.plot(x_midpoints, x_profile)
plt.title("Intensity Slice of I(x, y=0)")
plt.xlabel("x")
plt.ylabel("Intensity")

plt.tight_layout()
plt.show()

wave_numbers, spectrum = DiffusionAnalysis2D.compute_power_spectrum(y_profile)

plt.figure(figsize=(8, 5))
plt.plot(wave_numbers, spectrum, label="Power Spectrum")
plt.title("Power Spectrum of I(x=0, y) (1D)")
plt.xlabel("Wave Number (k)")
plt.ylabel("Power Spectrum")
plt.grid(True)
plt.legend()
plt.show()

analysis = DiffusionAnalysis2D(histogram=histogram, x_bins=x_bins, y_bins=y_bins)

autocorr_2d = analysis.compute_2d_autocorrelation()

analysis.plot_2d_autocorrelation(autocorr_2d)
