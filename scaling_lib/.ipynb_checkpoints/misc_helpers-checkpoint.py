import torch
import numpy as np
import tdwg.lib.ftutils_np as ftutils

def unitary_fidelity(U_target: torch.Tensor, U_num_inv: torch.Tensor) -> torch.Tensor:
    """
    Compute the fidelity between two unitary matrices.
    
    Args:
        U_target (torch.Tensor): Target unitary matrix of shape (N, N).
        U_num_inv (torch.Tensor): Approximate unitary matrix of shape (N, N).
    
    Returns:
        torch.Tensor: The fidelity value.
    """
    N = U_target.shape[0]
    fidelity = np.abs(np.trace(U_target.conj().T @ U_num_inv)) / N
    return fidelity


def smoothen_waveguide(x_axis, dn_wg_slice, kc):
    Nx = len(x_axis)
    dx = x_axis[1] - x_axis[0]
    k_axis = 2*np.pi * ftutils.ft_f_axis(Nx, dx)
    
    # Apply Fourier Transform
    dn_wg_slice_k = ftutils.fft_centered(dn_wg_slice)
    # Gaussian filter in k-space
    H_k = np.exp(-(k_axis / kc)**2)
    # Apply the filter
    dn_wg_slice_k_smooth = dn_wg_slice_k * H_k
    # Inverse Fourier Transform
    dn_wg_slice = ftutils.ifft_centered(dn_wg_slice_k_smooth).real

    return dn_wg_slice
    
def add_absorbing_boundary(x_axis, dn_slice, k0, abs_width=10, sigma_max=0.05, power=2):
    """
    Add a PML-like absorbing boundary to the index profile.
    """
    Nx = len(x_axis)
    sigma = np.zeros_like(x_axis)
    
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
    
def E2a(E_list, Evecs, x_axis):
    dx = x_axis[2]-x_axis[1]
    T = torch.tensor(Evecs.T, dtype=torch.cfloat)
    a_list = (E_list@T).cpu().numpy().T*dx; #ff for full field
    return a_list

def a2atilde(a_list, betas, z_list):
    atilde_list = [oi*np.exp(-1j*ev*z_list) for (oi, ev) in zip(a_list, betas)]
    return atilde_list