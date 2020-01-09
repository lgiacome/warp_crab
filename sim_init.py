import sys
import os
from warp import *

BIN = os.path.expanduser("../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from warp_pyecloud_sim_new import warp_pyecloud_sim
from chamber import RectChamber
from lattice_elements import Dipole

enable_trap = True

nz = 10
N_mp_max_slice = 6000
init_num_elecs_slice = 2*10**5
dh = 3.e-4
width = 2*35e-3
height = 2*18e-3
nx = 10 #int(np.ceil(width/dh))
ny = 10 #int(np.ceil(height/dh))

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 100
beta_y = 100

beam_gamma = 479.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
print(sigmax)


chamber = RectChamber(width, height, -0.5, 0.5)
lattice_elem = Dipole(-0.5, 0.5, 0.53)

kwargs = {'enable_trap': enable_trap,
	'z_length': 1.,
	'nx': nx,
	'ny': ny, 
	'nz': nz,
	'n_bunches': 50,
    'b_spac' : 25e-9,
    'beam_gamma': beam_gamma, 
	'sigmax': sigmax,
    'sigmay': sigmay, 
    'sigmat': 1.000000e-09/4.,
    'bunch_intensity': 1.1e11, 
    'init_num_elecs': init_num_elecs_slice*nz,
    'init_num_elecs_mp': int(0.7*N_mp_max_slice*nz), 
	'By': 0.53549999999999998,
    'pyecloud_nel_mp_ref': init_num_elecs_slice/(0.7*N_mp_max_slice),
	'dt': 25e-12,
    'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
    'chamber_type': 'rect', 
    'flag_save_video': True,
    'Emax': 332., 
    'del_max': 1.7,
    'R0': 0.7, 
    'E_th': 35, 
    'sigmafit': 1.0828, 
    'mufit': 1.6636,
    'secondary_angle_distribution': 'cosine_3D', 
    'N_mp_max': N_mp_max_slice*nz,
    'N_mp_target': N_mp_max_slice/3*nz,
	'flag_checkpointing': True,
	'checkpoints': np.linspace(1, 30, 30),
    'flag_output': True,
    'bunch_macro_particles': 1e5,
    't_offs': 2.5e-9,
    'width' : width,
    'height' : height,
    'output_filename': 'warp_out.mat',
    'flag_relativ_tracking': True,
    'lattice_elem': lattice_elem,
    'chamber': chamber
}

sim = warp_pyecloud_sim(**kwargs)

sim.all_steps()
