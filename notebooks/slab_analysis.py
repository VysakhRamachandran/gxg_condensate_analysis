import pytim
from pytim import ITIM
from pytim.observables import Profile
from pytim.datafiles import LJ_GRO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
from MDAnalysis import transformations
from scipy.optimize import curve_fit, fsolve
from scipy.optimize import least_squares

def _calc_zpatch(z, h):
    '''Internal function for slab_center used for patching z profile''' 
    cutoff, ct, ct_max = 0, 0, 0
    zwindow, hwindow, zpatch, hpatch = [], [], [], []
    
    for ix, x in enumerate(h):
        
        if x > cutoff:   
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
            
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow                
            ct = 0.0
            zwindow, hwindow= [], []
    
    return np.array(zpatch), np.array(hpatch)

def center_slab(pdbfile, dcdfile, name, start=None, end=None, step=10):
    '''
    This function processes dcd file and outputs trajectory with centered slab positions.
    
    After centereing load trajectory and make molecules whole again for visualization
    ---
    traj = md.load(dcdfile, top=pdbfile)
    traj.make_molecules_whole(inplace=True)
    '''

        
    u = mda.Universe(pdbfile, dcdfile)
    n_frames = len(u.trajectory[start:end:step])
    
    ag = u.atoms
    n_atoms = ag.n_atoms
    lz = u.dimensions[2]
    
    edges = np.arange(0, lz+1, 1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    
    hs = np.zeros((n_frames, n_bins))
    
    with mda.Writer(f'./{name}.dcd',n_atoms) as W:
        
        for t, ts in enumerate(u.trajectory[start:end:step]):
            
            # shift max density to center
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0, 0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = _calc_zpatch(z, h)
            if np.sum(hpatch) != 0:
                zmid = np.average(zpatch, weights=hpatch)
                ag.translate(np.array([0, 0, -zmid+0.5*lz]))
                ts = transformations.wrap(ag)(ts)

                zpos = ag.positions.T[2]
                h, e = np.histogram(zpos, bins=edges)
                hs[t] = h
                W.write(ag)
            else:
                continue
            
    np.save(f'{name}_hist.npy', hs, allow_pickle=False)
    
    return hs, (z-z.mean())/10.0

def density_profile(u, chain_L=3, nbins=90, start=0, period=100):
    '''Calculate density profile from equilibrated coexistence simulations 
    by dividing box into slabs and histograming particle counts along z, the longest dimension:
    
    #Example usage
    traj = gsd.hoomd.open('test.gsd', 'rb')
    binz, dens, dens_av = density_profile(traj, nbins=61, start=100, period=100)
    
    traj: A gsd trajectory
    sel_id: ids of particles to include up to e.g all particles iwth ids < 3
    chain_L: length of chain
    nbins: number of slabs to devidide the box
    start: snapshots to consider
    period: interval of snapshots 
    '''
    
    dens=[]
    
    if (nbins % 2) != 0:
        nbins += 1
    
    for ts in u.trajectory[start:-1:period]:
        
        Lx, Ly, Lz = ts.dimensions[0], ts.dimensions[1], ts.dimensions[2]
    
        dz = Lz/(nbins)
        vbin = Lx * Ly * dz 

        prot = u.select_atoms('protein')
    
        posz1 = [i[2] for i in prot.positions]
        posz1 = posz1 - (Lz/2)
        
        h1, binz = np.histogram(posz1, bins = nbins, range=(-Lz/2, Lz/2))
        
        rhoz =  h1 / chain_L #divide chain length -> molecule/bin
        rhoz /= vbin 
        rhoz *= 10**27 / 6.022e23 #convert to mol/l
        
        dens.append(rhoz) 
        
    density = np.array(dens)
        
    return (binz[1:]+binz[-1:]), density, density.mean(axis=0)

def tanh_fit(binz, dens_av):
    '''Plot the average density profile along with best fit to tanh function with rho1>rho2 for the two coexisting phases'''
            
    def tanh_func(z, rho1, rho2, delta, z0): 
        return 0.5*(rho1+rho2) - 0.5*(rho1-rho2) * np.tanh((z - z0) * 2.19722/delta)
    
    binz1, binz2 = np.split(binz,2)
    rhoz1, rhoz2 = np.split(dens_av, 2)
    rho_av1 = (rhoz1[::-1]+ rhoz2[:])/2
    
    plt.plot(binz1,  rho_av1, label='data', color='blue')
    
    popt, pcov = curve_fit(tanh_func, binz1, rho_av1, p0=[rho_av1.max(), rho_av1.min(), 1, binz1.max()/2])
    plt.plot(binz1, tanh_func(binz1, *popt), '--', color='red', label='best fit')
    
    plt.xlim([binz1.min(),    1.01*binz1.max()])
    plt.ylim([rho_av1.min(),  1.01*rho_av1.max()])
    plt.xlabel('z',fontsize=20)
    plt.ylabel(r'$\rho(z)$',fontsize=20)
    plt.legend()
    plt.title(fr' $\rho_1={popt[0]:.2f}, \rho_2={popt[1]:.2f}, \delta={popt[2]:.2f}, z_0={popt[3]:.2f}$')
    
    rho1 = popt[0]
    rho2 = popt[1]
    
    return rho1, rho2
    
    
    
    
    
