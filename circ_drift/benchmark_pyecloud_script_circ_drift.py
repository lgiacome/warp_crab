import sys
import os

BIN = os.path.expanduser("../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from  warp_pyecloud_dipole import warp_pyecloud_dipole

run_in_other_process = False

enable_trap = True

from run_in_separate_process import run_in_separate_process
nz = 100
N_mp_max_slice = 60000
init_num_elecs_slice = 2*10**5
dh = 3.e-4
radius = 23e-3
n_bunches = 50

nx = int(np.ceil(2*radius/dh))
ny = int(np.ceil(2*radius/dh))

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
kwargs = {'enable_trap': enable_trap,
	'z_length': 1.,
	'nx': nx,
	'ny': ny, 
	'nz': nz,
    'dh_t': dh, 
	'n_bunches': n_bunches,
    'b_spac' : 25e-9,
    'beam_gamma': beam_gamma, 
	'sigmax': sigmax,
    'sigmay': sigmay, 
    'sigmat': 1.000000e-09/4.,
    'bunch_intensity': 1.1e11, 
    'init_num_elecs': init_num_elecs_slice*nz,
    'init_num_elecs_mp': int(0.7*N_mp_max_slice*nz), 
	'By': 0.,
    'pyecloud_nel_mp_ref': init_num_elecs_slice/(0.7*N_mp_max_slice),
	'dt': 25e-12,
    'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
    'chamber_type': 'circle', 
    'flag_save_video': False,
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
	'checkpoints': np.linspace(1, n_bunches, n_bunches),
    'flag_output': True,
    'bunch_macro_particles': 1e5,
    't_offs': 2.5e-9,
    'output_name': 'warp_out_circle_drift.h5',
    'flag_relativ_tracking': True,
    'radius':  radius
}

if run_in_other_process:
  res = run_in_separate_process(warp_pyecloud_dipole, kwargs=kwargs)
else:
  res = warp_pyecloud_dipole(**kwargs)


