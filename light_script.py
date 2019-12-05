import numpy as np
from  warp_pyecloud_dipole import warp_pyecloud_dipole

run_in_other_process = False

enable_trap = True

from run_in_separate_process import run_in_separate_process

sey_params_dict={}
sey_params_dict['Emax'] = 300.
sey_params_dict['del_max'] = 1.8
sey_params_dict['R0'] = 0.7
sey_params_dict['E_th'] = 30.
sey_params_dict['sigmafit'] = 1.09
sey_params_dict['mufit'] = 1.66
sey_params_dict['secondary_angle_distribution'] = 'cosine_3D'


kwargs = {'enable_trap': enable_trap,
	'z_length': 1.,
	'nx': 5,
	'ny': 5, 
	'nz': 20, 
	'n_bunches': 10,
    'bunch_macro_particles': 1e4,
        'b_spac' : 25e-9,
        'beam_gamma': 479., 
	'sigmax': 2e-4,
        'sigmay': 2.1e-4, 
        'sigmat': 1.000000e-09/4.,
        'bunch_intensity': 1e11, 
        'init_num_elecs': 1.e5,
        'init_num_elecs_mp': 5*10**2, 
	'By': 0.53,
        'pyecloud_nel_mp_ref': 2e2,
	'dt': 25e-12,
        'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
        'chamber_type': 'rect',
        'width': 10e-2,
        'height': 10e-2,
        'flag_save_video': False,
        'Emax': 300., 
        'del_max': 2.5,
        'R0': 0.7, 
        'E_th': 30, 
        'sigmafit': 1.09, 
        'mufit': 1.66,
        'secondary_angle_distribution': 'cosine_3D', 
        'N_mp_max': 5*10**3,
        'N_mp_target': 5*10**1,
	'flag_checkpointing': True,
	'checkpoints': np.linspace(1,10,10),
    't_offs': 0,
    'flag_output': True
}

if run_in_other_process:
  res = run_in_separate_process(warp_pyecloud_dipole, kwargs=kwargs)
else:
  res = warp_pyecloud_dipole(**kwargs)


