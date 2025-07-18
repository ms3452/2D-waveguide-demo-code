import numpy as np
import torch
import scaling_lib.ftutils_torch as ftutils

class Waveguide():
    def __init__(self, neff, x_axis, z_axis, background_delta_n):
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
        
        self.fx_axis = ftutils.ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis.to(torch.complex64)

        self.x2ind = lambda x: int(np.argmin(np.abs(self.x_axis-x)))
        self.z2ind = lambda z: int(np.argmin(np.abs(self.z_axis-z)))
        self.zlist2ind = lambda z: int(np.argmin(np.abs(self.z_list-z)))

        if background_delta_n.shape != (self.Nx,):
            raise ValueError('spatial_map has wrong shape, should be [self.Nx,]')
        self.background_delta_n = background_delta_n