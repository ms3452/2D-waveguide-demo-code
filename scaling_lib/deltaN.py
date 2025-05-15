# This file will all of the codes that are associated with deltaN!
import numpy as np
import torch
from scipy.linalg import lstsq
from scipy.linalg import logm

# --- Start Simplified ---- #
def dn_crosssec(Evecs, x_axis):
    dx = x_axis[1] - x_axis[0]
    N_kl = 1/((Evecs**2)@(Evecs.T)**2 * dx)
    E_k_E_l = Evecs[:, np.newaxis, :] * Evecs[np.newaxis, :, :]
    return N_kl[:,:,np.newaxis] * E_k_E_l

def ReAexp(A, betas, z_axis):
    b_kl = betas[:, np.newaxis] - betas[np.newaxis, :]
    phase = b_kl[:,:,np.newaxis] * z_axis[np.newaxis, np.newaxis, :]
    return np.real(A[:,:,np.newaxis] * np.exp(1j*phase))

def make_deltaN_exp(A, Evecs, betas, x_axis, z_axis):
    x_dependence = dn_crosssec(Evecs, x_axis)
    z_dependence = ReAexp(A, betas, z_axis)
    dn_kl = x_dependence[:,:,np.newaxis,:] * z_dependence[:,:,:,np.newaxis] 
    return np.sum(dn_kl, (0,1))
# --- End Simplified ---- #

def make_deltaN_crosssec_kl(k, l, Evecs, x_axis):
    """
    This is going to output N_kl*E_k(x)*E_l(x), which is the part of the DeltaN that is a crosssection!
    Inputs:
    k, l: The index of the modes that will be coupled; so they are integers.
    Evec: the modes of the waveguides, which must be real.
    """
    dx = x_axis[1] - x_axis[0]
    Δn_crosssec_kl = Evecs[k]*Evecs[l]
    Δn_crosssec_kl /= np.sum(Δn_crosssec_kl**2)*dx #divide by N here.
    return Δn_crosssec_kl

def make_deltaN_dc(A_diag, Evecs, x_axis, z_axis):
    dx = x_axis[1] - x_axis[0]
    Nz = len(z_axis)
    
    mat = np.abs(Evecs)**2*dx
    deltaN_dc_crosssec = lstsq(mat, A_diag)[0]
    return np.tile(deltaN_dc_crosssec, (Nz, 1))

# Add the function here when the time is right!
# Probably best to go slower and even have another function!

# Now make yet another function!
def make_deltaN_kl_Re(k, l, Evecs, betas, x_axis, z_axis):
    """
    Returns deltaN_prog_kl_Re in the main manuscript. 
    Note that I got rid of the prog part, to make the code more readable. 

    So this does not have any coefficients!
    """
    Nz = len(z_axis) #principle of passing less variables, if they can rederived in an inner function!
    deltaN_crosssec_kl = make_deltaN_crosssec_kl(k, l, Evecs, x_axis)
    deltaN_crosssec_kl_tiled = np.tile(deltaN_crosssec_kl, (Nz, 1)).T #This is basically gonna repeat the same thing across the other dimension.
    return (deltaN_crosssec_kl_tiled*np.cos((betas[k]-betas[l])*z_axis)).T #this is so that the format is given by [Nz, Nx] what the simulator takes in!

# Now make yet another function!
def make_deltaN_kl_Im(k, l, Evecs, betas, x_axis, z_axis):
    """
    Returns deltaN_prog_kl_Re in the main manuscript. 
    Note that I got rid of the prog part, to make the code more readable. 

    So this does not have any coefficients!
    """
    Nz = len(z_axis) #principle of passing less variables, if they can rederived in an inner function!
    deltaN_crosssec_kl = make_deltaN_crosssec_kl(k, l, Evecs, x_axis)
    deltaN_crosssec_kl_tiled = np.tile(deltaN_crosssec_kl, (Nz, 1)).T #This is basically gonna repeat the same thing across the other dimension.
    return (deltaN_crosssec_kl_tiled*np.sin((betas[l]-betas[k])*z_axis)).T #this is so that the format is given by [Nz, Nx] what the simulator takes in!

def make_deltaN(A, Evecs, betas, x_axis, z_axis):
    Nmodes = len(betas)
    Nz = len(z_axis) #principle of passing less variables, if they can rederived in an inner function!
    Nx = len(x_axis) #principle of passing less variables, if they can rederived in an inner function!

    deltaN = make_deltaN_dc(np.diag(A), Evecs, x_axis, z_axis)
    for k in range(Nmodes):
        for l in range(Nmodes):
            if k is not l:
                 deltaN += np.real(A[k, l])*make_deltaN_kl_Re(k, l, Evecs, betas, x_axis, z_axis)
                 deltaN += np.imag(A[k, l])*make_deltaN_kl_Im(k, l, Evecs, betas, x_axis, z_axis)
    return deltaN
    
# Now make yet another function!
def make_deltaN_kl_exp(k, l, Evecs, betas, x_axis, z_axis):
    """"Uses simplified expression from Martin"""
    Nz = len(z_axis) #principle of passing less variables, if they can rederived in an inner function!
    deltaN_crosssec_kl = make_deltaN_crosssec_kl(k, l, Evecs, x_axis)
    deltaN_crosssec_kl_tiled = np.tile(deltaN_crosssec_kl, (Nz, 1)).T #This is basically gonna repeat the same thing across the other dimension.
    return (deltaN_crosssec_kl_tiled*np.exp(1j*(betas[k]-betas[l])*z_axis)).T #this is so that the format is given by [Nz, Nx] what the simulator takes in!
    
def make_deltaN(A, Evecs, betas, x_axis, z_axis):
    Nmodes = len(betas)
    Nz = len(z_axis) #principle of passing less variables, if they can rederived in an inner function!
    Nx = len(x_axis) #principle of passing less variables, if they can rederived in an inner function!

    deltaN = make_deltaN_dc(np.diag(A), Evecs, x_axis, z_axis)
    for k in range(Nmodes):
        for l in range(Nmodes):
            if k is not l:
                 deltaN += np.real(A[k, l])*make_deltaN_kl_Re(k, l, Evecs, betas, x_axis, z_axis)
                 deltaN += np.imag(A[k, l])*make_deltaN_kl_Im(k, l, Evecs, betas, x_axis, z_axis)
    return deltaN

def U2A(U, k0, Lz):
    return logm(U)/(1j*k0*Lz)




