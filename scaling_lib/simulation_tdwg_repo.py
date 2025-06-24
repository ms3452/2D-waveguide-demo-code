"""
This is the simulation code.
Note that this is an old commit from Sat Jul 22 12:24:17 2023
Ironically, this version is better because it doesn't have too much advanced features...
"""

import torch
import torch.fft as fft
from fft_conv_pytorch import fft_conv, FFTConv2d
import numpy as np
from scipy.stats import multivariate_normal
import scaling_lib.ftutils_np as ftutils
import matplotlib.pyplot as plt
import astropy.units as u
import copy
from scipy.interpolate import interp1d, interp2d

class WaveguideSimulationWithBackground():
    def __init__(self, neff, x_axis, z_axis, Ncom=1, fresnel = False):
        #all the units for length are in microns!
        self.n =  neff # effective refractive index for the slab waveguide
        
        self.z_axis = z_axis
        self.Nz = len(z_axis)
        self.dz = z_axis[1] - z_axis[0]
        self.Lz = z_axis.max()
        
        self.x_axis = x_axis
        self.Nx = len(x_axis)
        self.dx = x_axis[1] - x_axis[0]
        
        self.lam0 = 1.55 #wavelength of the fundamental
        self.k0 = 2*np.pi/self.lam0 # k-number in free space
        
        self.Ncom = Ncom # wavefront is saved every Ncom integration steps

        self.fx_axis = ftutils.ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis.astype(complex)

        # The following defines *dimensionless* quantities, which are used in the simulation internal loop!
        if fresnel: self.phase_shift = self.kx_axis**2/(2*self.n*self.k0)*self.dz # Fresnel
        if not fresnel: self.phase_shift = -self.dz * np.sqrt(self.n**2 * self.k0**2 - self.kx_axis**2) + self.dz * self.n * self.k0 # extended Fresnel
        difr_list = np.fft.fftshift(np.exp((-1j*self.phase_shift)))
        difr_half_list = np.fft.fftshift(np.exp((-1j*self.phase_shift/2)))
        self.difr_list = torch.tensor(difr_list)
        self.difr_half_list = torch.tensor(difr_half_list)
        self.k0dz = (self.k0*self.dz)

        self.x2ind = lambda x: np.argmin(np.abs(self.x_axis-x))
        self.z2ind = lambda z: np.argmin(np.abs(self.z_axis-z))
        self.zlist2ind = lambda z: np.argmin(np.abs(self.z_list-z))

        
    def set_background_delta_n(self, background_delta_n):
        if background_delta_n.shape != (self.Nx,):
            raise ValueError('spatial_map has wrong shape, should be [self.Nx,]')
        self.background_delta_n = background_delta_n

    def run_simulation(self, a, delta_n):
        """
        Use this if you want code to be fast!
        a: The input beam!
        """
        ### Strang splitting
        difr_list = self.difr_list.to(a.device)
        difr_half_list = self.difr_half_list.to(a.device)
        delta_n_term = torch.exp(1j*self.k0dz*(self.background_delta_n + delta_n)) #takes about 40us to run, so don't worry about optimizing it!
        
        ak = fft.fft(a)
        ak = difr_half_list * ak
        for delta_n_term_slice in delta_n_term:
            a = fft.ifft(ak)
            a = delta_n_term_slice * a
            ak = fft.fft(a)
            ak = difr_list * ak
        ak = difr_half_list.conj() * ak
        a = fft.ifft(ak)
        return a
        
    def run_simulation_slow(self, a, delta_n = None):
        """
        Around 2X slower than the fast version for Nz being 1000
        # Characeterize in more detail for other Nz
        """
        self.a_list = []
        self.ak_list = []
        self.z_list = []

        if delta_n is None: delta_n = self.delta_n
        else: self.delta_n = delta_n
            
        difr_list = self.difr_list.to(a.device)
        difr_half_list = self.difr_half_list.to(a.device)
        delta_n_term = torch.exp(1j*self.k0dz*(self.background_delta_n + delta_n)) #takes about 40us to run, so don't worry about optimizing it!

        z = 0
        ak = fft.fft(a)
        ak = difr_half_list * ak
        for (z_ind, delta_n_term_slice) in enumerate(delta_n_term):
            a = fft.ifft(ak)
            a = delta_n_term_slice * a
            ak = fft.fft(a)
            ak = difr_list * ak
            z += self.dz

            if (z_ind + 1) % self.Ncom == 0:
                self.a_list.append(a)
                self.ak_list.append(torch.fft.ifftshift(ak))
                self.z_list.append(copy.deepcopy(z))
        ak = difr_half_list.conj() * ak
        a = fft.ifft(ak)

        self.z_list = torch.tensor(self.z_list)
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
    
    def _plot_delta_n(self, xlim=200):
        wg = self
        plt.pcolormesh(wg.z_axis/1000, wg.x_axis, wg.delta_n.T.detach().cpu()*1e3, cmap="binary", shading="auto")
        plt.colorbar()
        plt.ylabel("x (um)")
        plt.xlabel("z (mm)")
        plt.ylim(-xlim, xlim)
        plt.gca().invert_xaxis()
        plt.title(r"$\Delta n\ \  (10^{-3})$")
        plt.grid(alpha=0.5)

    def _plot_Imat_x(self, xlim=200, renorm_flag=True):
        wg = self
        Emat_x = wg.Emat_x.detach().cpu().numpy()
        Imat_x = np.abs(Emat_x)**2

        if renorm_flag:
            Imat_x = (Imat_x.T/np.max(Imat_x, axis=1)).T


        plt.pcolormesh(wg.z_list/1000, wg.x_axis, Imat_x.T, cmap="inferno", vmin=0, shading="auto")
        plt.xlabel("z (mm)")
        plt.ylabel("x (um)")
        plt.ylim(-xlim, xlim)
        plt.gca().invert_xaxis()
        plt.title("Spatial intensity")
        plt.grid(alpha=0.5)

    def _plot_Imat_f(self, flim=80, renorm_flag=True):
        wg = self
        Emat_f = wg.Emat_f.detach().cpu().numpy()
        Imat_f = np.abs(Emat_f)**2

        if renorm_flag:
            Imat_f = (Imat_f.T/np.max(Imat_f, axis=1)).T

        plt.pcolormesh(wg.z_list/1000, wg.fx_axis*1000, Imat_f.T, cmap="inferno", vmin=0, shading="auto")
        plt.xlabel("z (mm)")
        plt.ylabel("f (1/mm)")
        plt.ylim(-flim, flim)
        plt.gca().invert_xaxis()
        plt.title("Wavevector intensity")
        plt.grid(alpha=0.5)

    def plot_mats(self, xlim=200, flim=80, renorm_flag=True):
        """
        renorm_flag: If true, the plots are renormalized to the maximum value of the intensity for a given zaxis point.

        TODO list
        - Add the title for the different plots
        - Make the colorbar for the deltaN more nicely formatted to save space
        """        
        wg = self #to save rewriting the code!


        fig, axs = plt.subplots(1, 3, figsize=(12, 2))
        fig.subplots_adjust(wspace=0.4)

        plt.sca(axs[0])
        self._plot_delta_n(xlim=xlim)

        plt.sca(axs[1])
        self._plot_Imat_x(xlim=xlim, renorm_flag=renorm_flag)

        plt.sca(axs[2])
        self._plot_Imat_f(flim=flim, renorm_flag=renorm_flag)

        for ax in axs:
            plt.sca(ax)

        return fig, axs