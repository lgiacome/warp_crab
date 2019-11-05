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
	'nx': 16,
	'ny': 16, 
	'nz': 150, 
	'n_bunches': 10,
        'b_spac' : 25e-9,
        'beam_gamma': 479., 
	'sigmax': 2e-4,
        'sigmay': 2.1e-4, 
        'sigmat': 1.000000e-09/4.,
        'bunch_intensity': 1e11, 
        'init_num_elecs': 1.e8,
        'init_num_elecs_mp': 10**5, 
	'By': 0.53,
        'pyecloud_nel_mp_ref': 1e3,
	'dt': 25e-12,
        'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
        'chamber_type': 'rect', 
        'flag_save_video': False,
        'Emax': 300., 
        'del_max': 1.8,
        'R0': 0.7, 
        'E_th': 30, 
        'sigmafit': 1.09, 
        'mufit': 1.66,
        'secondary_angle_distribution': 'cosine_3D', 
        'N_mp_max': 10**8,
        'N_mp_target': 10**5
}

if run_in_other_process:
  res = run_in_separate_process(warp_pyecloud_dipole, kwargs=kwargs)
else:
  res = warp_pyecloud_dipole(**kwargs)


