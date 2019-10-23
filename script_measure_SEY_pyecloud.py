import numpy as np
from  measure_SEY_pyecloud import measure_SEY

enable_trap = True
ene_array = np.linspace(5,1500,30)

sey_curve = np.zeros_like(ene_array)
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


for ii, ene in enumerate(ene_array):
    print(ii)
    sys.stdout = text_trap
    res = run_in_separate_process(measure_SEY, [ene, Nmp, N_elec_p_mp, sey_params_dict])
    sey_curve[ii] = res['SEY']
    sys.stdout = original

import matplotlib.pyplot as plt
plt.close('all')

plt.plot(ene_array,sey_curve)

from PyECLOUD.sec_emission_model_ECLOUD import yield_fun2

del_ref, _ = yield_fun2(E=ene_array, costheta=1., s=1.35, E0=150,
    Emax=sey_params_dict['Emax'],
    del_max=sey_params_dict['del_max'],
    R0=sey_params_dict['R0'])

plt.plot(ene_array, del_ref, 'g')

plt.show()

