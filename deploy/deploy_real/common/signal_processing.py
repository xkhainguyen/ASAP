from scipy.signal import butter, filtfilt

def high_pass_filter(data, cutoff=0.1, fs=100):
    """ Apply a high-pass filter to remove low-frequency drift. """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def low_pass_filter(data, cutoff=5.0, fs=100):
    """
    Apply a low-pass filter to smooth the input signal.
    
    Parameters:
    - data (array-like): The input signal to be filtered.
    - cutoff (float): Cutoff frequency in Hz (default=5.0).
    - fs (float): Sampling frequency in Hz (default=100).
    
    Returns:
    - Filtered signal (same shape as input).
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(1, normal_cutoff, btype='low', analog=False)  # 1st-order low-pass filter
    return filtfilt(b, a, data, axis=0)  # Apply zero-phase filtering

def rk4_integrate(accel, velocity, position, dt):
    """ Integrate acceleration using RK4 for better stability. """
    k1 = accel
    k2 = accel + 0.5 * dt * k1
    k3 = accel + 0.5 * dt * k2
    k4 = accel + dt * k3
    new_velocity = velocity + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    new_position = position + dt * new_velocity
    return new_velocity, new_position
