from scaling_lib.simulation_tdwg_repo import WaveguideSimulationWithBackground
import torch.nn as nn
import torch
import numpy as np
import tdwg.lib.pnn_utils as pnn_utils 
import torch.optim as optim
from tdwg.lib.misc_utils import timestring


class TDwgNet(nn.Module):
    def __init__(self, wg, delta_n_val, background_dn, Evecs, betas, dx, k_cprog, device):
        super(TDwgNet, self).__init__()
        mod = 0.5*torch.ones([wg.Nz, wg.Nx], requires_grad=True, dtype=torch.float32).to(device)
        self.mod = pnn_utils.Parameter(mod, limits=[0.05, 0.95]) #ok this should still be there...
        self.background_dn = background_dn.to(device)
        self.delta_n_val = delta_n_val
        self.input_beams = torch.from_numpy(Evecs).to(dtype=torch.complex128).to(device)
        self.output_modes = torch.from_numpy(Evecs).to(dtype=torch.complex128).to(device)*dx
        self.wg = wg
        self.betas = torch.tensor(betas, dtype=torch.float32).to(device)
        self.k_cprog = k_cprog

    def forward(self, indices = None, fast_flag=True, ind=0):
        mod = self.mod
        mod = mod.clip(0, 1)
        self.wg.set_background_delta_n(self.background_dn)
        
        if fast_flag:
            output_beams = self.wg.run_simulation(self.input_beams[indices], self.delta_n_val*self.smoothen(mod))
        else:
            output_beams = self.wg.run_simulation_slow(self.input_beams[ind, :])
        
        a_out = output_beams@self.output_modes.T
        U = (a_out.T*torch.exp(-1j*self.betas[indices]*self.wg.Lz)).T
        return U
        
    def smoothen(self, mod):
        # self.mod: (Nz, Nx)
        Nz, Nx = mod.shape
        dz = self.wg.z_axis[1] - self.wg.z_axis[0]
        dx = self.wg.x_axis[1] - self.wg.x_axis[0]
    
        # Frequency axes
        kz = torch.fft.fftfreq(Nz, d=dz).to(self.mod.device) * 2 * torch.pi  # angular frequency
        kx = torch.fft.fftfreq(Nx, d=dx).to(self.mod.device) * 2 * torch.pi
    
        # Create 2D meshgrid for frequencies
        Kz, Kx = torch.meshgrid(kz, kx, indexing='ij')
    
        # Construct Gaussian filter: exp(- (k^2 / self.k_cprog^2))
        gaussian_filter = torch.exp(-((Kz**2 + Kx**2) / (self.k_cprog**2)))
    
        # FFT of the mod parameter
        mod_fft = torch.fft.fft2(mod)
    
        # Apply the filter
        mod_fft_filtered = mod_fft * gaussian_filter
    
        # Inverse FFT to get back to spatial domain
        mod_filtered = torch.fft.ifft2(mod_fft_filtered).real
    
        return mod_filtered
    
def L2(p, q):
    x = torch.abs(p-q)**2
    return x.sum(dim = -1).mean()/2

# def run_training_loop(tdwg_pnn, U_target, iterations, lr, gamma=0.99, print_interval=1):
#     optimizer = optim.Adam(tdwg_pnn.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
#     loss_list = []

#     tdwg_pnn.train()
#     for i in range(iterations):
#         optimizer.zero_grad()
#         U_num = tdwg_pnn.forward()
#         l_objective = L2(U_target, U_num)

#         l_lagrange = pnn_utils.lagrangian(tdwg_pnn, lag_amp = 1, factor = 20)
#         l = l_objective + l_lagrange 
#         # l = l_objective
#         l.backward()
#         optimizer.step()
#         scheduler.step()
#         loss_list.append(l.detach().cpu().data)

#         if i % print_interval == 0:
#             print(f"Iteration: {i}, Loss: {l.detach().cpu().data:.5f}")
#         if has_converged(loss_list):
#             print(f"Converged at iteration: {i}, Loss: {l.detach().cpu().data:.5f}")
#             break
#     return loss_list
def run_training_loop(tdwg_pnn, U_target, iterations, lr, batch_size, device, gamma=0.99, print_interval=1):
    optimizer = optim.Adam(tdwg_pnn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    U_target = U_target.to(device)
    loss_list = []
    
    n = len(tdwg_pnn.input_beams)  # length of the original vector
    indices = np.arange(n)
    batch_indices = np.array_split(indices, np.ceil(n / batch_size))  # Split into subvectors

    tdwg_pnn.train()
    for i in range(iterations):
        optimizer.zero_grad()

        for indices in batch_indices:
            U_num = tdwg_pnn.forward(indices)
            l_objective = L2(U_target[indices], U_num)
    
            l_lagrange = pnn_utils.lagrangian(tdwg_pnn, lag_amp = 1, factor = 20)
            l = l_objective + l_lagrange 
            # l = l_objective
            l.backward()
            
        optimizer.step()
        scheduler.step()
        loss_list.append(l.detach().cpu().data)

        if i % print_interval == 0:
            print(timestring() + f"--Iteration: {i}, Loss: {l.detach().cpu().data:.5f}")
            # get_cpu_memory_usage(threshold_gb = 0.01)
            # get_gpu_memory_usage(threshold_gb = 0.01)
        if has_converged(loss_list):
            print(timestring() + f"Converged at iteration: {i}, Loss: {l.detach().cpu().data:.5f}")
            break
    return loss_list


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