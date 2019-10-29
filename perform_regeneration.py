import numpy as np
from numpy.random import rand
from warp import echarge as charge
from warp import emass as mass

def perform_regeneration(target_N_mp, x_mp, y_mp, z_mp, vx_mp, vy_mp, vz_mp, nel_mp):
    N_mp = len(x_mp)
    if N_mp > target_N_mp:
        chrg = np.sum(nel_mp)
        erg = np.sum(0.5 / np.abs(charge / mass) * nel_mp[0:N_mp] * (vx_mp[0:N_mp] * vx_mp[0:N_mp] +  vy_mp[0:N_mp] * vy_mp[0:N_mp] + vz_mp[0:N_mp] * vz_mp[0:N_mp]))
        
        new_nel_mp_ref = chrg / target_N_mp
                    
        print 'Start SOFT regeneration. N_mp=%d Nel_tot=%1.2e En_tot=%1.2e'%(N_mp, chrg, erg)
                    
        #set_nel_mp_ref(new_nel_mp_ref)
                    
        death_prob = float(N_mp - target_N_mp) / float(N_mp)
                    
        flag_keep = np.array(len(x_mp) * [False])
        flag_keep[:N_mp] = (rand(N_mp) > death_prob)
        N_mp = np.sum(flag_keep)

	x_mp_new = np.zeros(N_mp)
	y_mp_new = np.zeros(N_mp)
	z_mp_new = np.zeros(N_mp)
	vx_mp_new = np.zeros(N_mp)
	vy_mp_new = np.zeros(N_mp)
	vz_mp_new = np.zeros(N_mp)
	nel_mp_new = np.zeros(N_mp)
                  
        x_mp_new[0:N_mp] = np.array(x_mp[flag_keep].copy())
        y_mp_new[0:N_mp] = np.array(y_mp[flag_keep].copy())
        z_mp_new[0:N_mp] = np.array(z_mp[flag_keep].copy())
        vx_mp_new[0:N_mp] = np.array(vx_mp[flag_keep].copy())
        vy_mp_new[0:N_mp] = np.array(vy_mp[flag_keep].copy())
        vz_mp_new[0:N_mp] = np.array(vz_mp[flag_keep].copy())
        nel_mp_new[0:N_mp] = np.array(nel_mp[flag_keep].copy())
                    
           
        chrg_before = chrg
        chrg_after = np.sum(nel_mp_new)
                    
        correct_fact = chrg_before / chrg_after
                    
        print 'Applied correction factor = %e'%correct_fact
                   
        nel_mp_new[0:N_mp] = nel_mp_new[0:N_mp] * correct_fact
                    
        chrg = np.sum(nel_mp_new)
        erg = np.sum(0.5 / np.abs(charge / mass) * nel_mp_new[0:N_mp] * (vx_mp_new[0:N_mp] * vx_mp_new[0:N_mp] + vy_mp_new[0:N_mp] * vy_mp_new[0:N_mp] + vz_mp_new[0:N_mp] * vz_mp_new[0:N_mp]))
                
        print 'Done SOFT regeneration. N_mp=%d Nel_tot=%1.2e En_tot=%1.2e'%(N_mp, chrg, erg)
    return x_mp_new,y_mp_new,z_mp_new,vx_mp_new,vy_mp_new,vz_mp_new,nel_mp_new,N_mp
