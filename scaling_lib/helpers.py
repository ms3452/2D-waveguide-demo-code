import torch
import time
import numpy as np

def has_converged(loss_list, window=5, std_threshold=0.01):
    """
    Check if training has converged based on the standard deviation of recent loss values.

    Args:
        loss_list (list): List of loss values per epoch.
        window (int): Number of recent epochs to consider.
        std_threshold (float): Maximum acceptable standard deviation.

    Returns:
        bool: True if loss has converged, False otherwise.
    """
    if len(loss_list) < window:
        return False  # Not enough data

    recent_losses = loss_list[-window:]
    std_dev = np.std(recent_losses)
    mean = np.mean(recent_losses)
    std_dev_rel = std_dev / mean

    return std_dev_rel < std_threshold  # Converged if std deviation is very small

def L2(p, q):
    x = torch.abs(p-q)**2
    return x.sum(dim = -1).mean()/2

def timestring():
    return time.strftime("%Y-%m-%d--%H-%M-%S")

def add_absorbing_boundary(x_axis, dn_slice, k0, abs_width=10, sigma_max=0.05, power=2):
    """
    Add a PML-like absorbing boundary to the index profile.
    """
    Nx = len(x_axis)
    sigma = torch.zeros_like(x_axis)
    
    x_min = x_axis[0]
    x_max = x_axis[-1]
    
    # Left boundary
    left_mask = x_axis <= x_min + abs_width
    x_left = x_axis[left_mask]
    sigma[left_mask] = sigma_max * ((x_min + abs_width - x_left) / abs_width) ** power

    # Right boundary
    right_mask = x_axis >= x_max - abs_width
    x_right = x_axis[right_mask]
    sigma[right_mask] = sigma_max * ((x_right - (x_max - abs_width)) / abs_width) ** power

    # Final complex index
    n_complex = dn_slice + 1j * sigma
    return n_complex

def gaussian(x, mu, sigma):
    mode = torch.exp(-(x-mu)**2 / 2/sigma**2)
    return mode / torch.sqrt((mode.abs()**2).sum())

def smoothen1d(x_axis, tensor, scale):
    Nx = len(x_axis)
    dx = x_axis[1] - x_axis[0]

    # Frequency axes
    fx = torch.fft.fftfreq(Nx, d=dx).to(tensor.device)

    # Construct Gaussian filter: exp(- (k^2 * scale^2))
    gaussian_filter = torch.exp(- fx**2 * scale**2 ) 

    # FFT of the tensor
    tensor_fft = torch.fft.fft(tensor)

    # Apply the filter
    tensor_fft_filtered = tensor_fft * gaussian_filter

    # Inverse FFT to get back to spatial domain
    tensor_filtered = torch.fft.ifft(tensor_fft_filtered)#.real

    return tensor_filtered