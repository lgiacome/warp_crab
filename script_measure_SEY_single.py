import numpy as np
from  measure_SEY_pyecloud import measure_SEY

run_in_other_process = True
enable_trap = False

ene = 5.

from run_in_separate_process import run_in_separate_process
import sys
original = sys.stdout
from io import BytesIO as StringIO
text_trap = {True: StringIO(), False: sys.stdout}[enable_trap]
Nmp = 10000
N_elec_p_mp = 1

sey_params_dict={}
sey_params_dict['Emax'] = 300.
sey_params_dict['del_max'] = 1.8
sey_params_dict['R0'] = 0.7
sey_params_dict['E_th'] = 30.
sey_params_dict['sigmafit'] = 1.09
sey_params_dict['mufit'] = 1.66
sey_params_dict['secondary_angle_distribution'] = 'cosine_3D'

kwargs = {
        'Ekin': ene, 
        'Nmp': Nmp,
        'N_elec_p_mp': N_elec_p_mp, 
        'sey_params_dict': sey_params_dict,
        'flag_video':False 
    }
sys.stdout = text_trap

if run_in_other_process:
    res = run_in_separate_process(measure_SEY, kwargs=kwargs)
else:
    res = measure_SEY(**kwargs)

sey_curve = res['SEY']
sys.stdout = original


from PyECLOUD.sec_emission_model_ECLOUD import yield_fun2

del_ref, _ = yield_fun2(E=np.array([ene]), costheta=1., s=1.35, E0=150,
    Emax=sey_params_dict['Emax'],
    del_max=sey_params_dict['del_max'],
    R0=sey_params_dict['R0'])


import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
sp1 = fig1.add_subplot(111)
sp1.plot(res['n_secelecs'], '.-')
sp1.plot(res['n_primelecs'], '.-')
plt.show()
