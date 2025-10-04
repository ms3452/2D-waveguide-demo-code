"""
Split-step Fourier method solver for wave propagation.

This module provides split-step solvers for simulating wave propagation in 2D waveguides:
- Generic split-step solver with user-defined Fourier and real-space operators
- Beam Propagation Method (BPM) solver inherited from the generic split-step solver
- Can store or not store internal fields for propagation visualization or memory-efficiency, respectively

Uses PyTorch for GPU-accelerated computations.
"""

import torch
import torch.fft as fft
import numpy as np
import tdwg_lib.ftutils_torch as ftutils

class SplitStepSolver():
    def __init__(self, D_step, D_half_step = None, Ncom = 1):
        self.D_step = D_step
        if D_half_step is None:
            D_half_step = torch.sqrt(D_step)
        self.D_half_step = D_half_step
        self.Ncom = Ncom
    
    def run_simulation(self, a, N, monitor = False):
        if monitor: 
            self.a_list = []
            self.ak_list = []

        D_step = self.D_step.to(a.device)
        D_half_step = self.D_half_step.to(a.device)
        N = N.to(a.device)
            
        ak = fft.fft(a)
        ak = D_half_step * ak
        for (z_ind, N_step) in enumerate(N):
            a = fft.ifft(ak)
            a = N_step * a
            ak = fft.fft(a)
            ak = D_step * ak

            if monitor:
                if (z_ind + 1) % self.Ncom == 0:
                    self.a_list.append(a)
                    self.ak_list.append(torch.fft.ifftshift(ak))
        ak = D_half_step.conj() * ak        
        a = fft.ifft(ak)

        if monitor:
            assert len(self.a_list) > 0, "No fields were stored—check Ncom or N length."
            self.Emat_x = torch.stack(self.a_list)
            self.Emat_f = torch.stack(self.ak_list)
    
            self.Eout_x = self.Emat_x[-1].detach().cpu().numpy()
            self.Eout_f = self.Emat_f[-1].detach().cpu().numpy()
            self.Iout_x = abs(self.Emat_x[-1].detach().cpu().numpy())**2
            self.Iout_f = abs(self.Emat_f[-1].detach().cpu().numpy())**2
    
            self.Ein_x = self.Emat_x[0].detach().cpu().numpy()
            self.Ein_f = self.Emat_f[0].detach().cpu().numpy()
            self.Iin_x = abs(self.Emat_x[0].detach().cpu().numpy())**2
            self.Iin_f = abs(self.Emat_f[0].detach().cpu().numpy())**2
        return a    


class BPMSplitStepSolver(SplitStepSolver):
    def __init__(self, x_axis, z_axis, n_ref, Ncom=1, k0=2*np.pi/1.55):
        self.k0 = k0  # Wave number (2π/λ)
        self.n_ref = n_ref
        
        self.x_axis = x_axis
        self.Nx = len(x_axis)
        self.dx = x_axis[1] - x_axis[0]

        self.z_axis = z_axis
        self.dz = z_axis[1] - z_axis[0]
        
        self.Ncom = Ncom # wavefront is saved every Ncom integration steps

        self.fx_axis = ftutils.ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis.to(torch.complex64)
        
        self.phase_shift = self.kx_axis**2/(2*self.n_ref*self.k0)*self.dz # Fresnel
        D_step = torch.fft.fftshift(torch.exp((-1j*self.phase_shift)))
        D_half_step = torch.fft.fftshift(torch.exp((-1j*self.phase_shift/2)))
        
        super().__init__(D_step, D_half_step, Ncom)

    def run_simulation(self, a, delta_n, monitor=False):
        """
        delta_n: tensor of shape (Nz, Nx) or (Nz, 1) or (Nz,)
        a: initial field (1D complex tensor)
        monitor: whether to use store intermediate values
        """
        device = a.device
        delta_n = delta_n.to(device)
        k0 = self.k0
        dz = self.dz.to(device=device).to(dtype=delta_n.dtype)

        # Compute nonlinear operator: N[z, x] = exp(i k0 delta_n dz)
        phase = 1j * k0 * delta_n * dz
        N = torch.exp(phase)

        return super().run_simulation(a, N, monitor)