import numpy as np
from numpy.random import rand
from warp import echarge as charge
from warp import emass as mass
from warp import top
#from warp.particles.Secondaries import Secondaries

def perform_regeneration(target_N_mp, wsp, Sec):
    # Just defining a shortcut
    # Old number of MPs
    N_mp = wsp.getn()
    old_N_mp = N_mp
    if N_mp > target_N_mp:
	# Compute old total charge and Ekin
        chrg = np.sum(wsp.getw())
        erg = np.sum(0.5 / np.abs(charge / mass) * wsp.getw()[0:N_mp] * (wsp.getvx()[0:N_mp] * wsp.getvx()[0:N_mp] + wsp.getvy()[0:N_mp] * wsp.getvy()[0:N_mp] + wsp.getvz()[0:N_mp] * wsp.getvz()[0:N_mp]))
        

        new_nel_mp_ref = chrg / target_N_mp  
        Sec.set_nel_mp_ref(new_nel_mp_ref)
	print 'New nel_mp_ref = %d'%(Sec.pyecloud_nel_mp_ref)

        print 'Start SOFT regeneration. N_mp=%d Nel_tot=%1.2e En_tot=%1.2e'%(N_mp, chrg, erg)
                    

	# Compute the death probability         
        death_prob = float(N_mp - target_N_mp) / float(N_mp)
	# Decide which MPs to keep/discard
        flag_keep = np.array(N_mp * [False])
        flag_keep[:N_mp] = (rand(N_mp) > death_prob)
	           
        # Compute the correction factor	
	chrg_before = chrg
        chrg_after = np.sum(wsp.getw()[flag_keep])
        correct_fact = chrg_before / chrg_after

	# Compute the indices of the MPs in pgroup.pid
        i_init = top.pgroup.ins[wsp.getjs()] - 1 
	inds_clear = np.where(flag_keep == False)[0] + i_init
	inds_keep = np.where(flag_keep == True)[0] + i_init

	# Rescale the survivors
	for ii in inds_keep:
		# Not clear why I am using wpid-1, but it seems to work 
       		top.pgroup.pid[ii,top.wpid-1] = top.pgroup.pid[ii,top.wpid-1]*correct_fact
	
	# Kill the discarded MPs
	for ii in inds_clear:
                top.pgroup.gaminv[ii] = 0.

	# Warning: top.clearpart is a Fortran routine so it has one-based ordering..
	top.clearpart(top.pgroup, wsp.getjs()+1, 1)        
	# Compute new number of MPs
	N_mp = wsp.getn()

        print 'Applied correction factor = %e'%correct_fact
                   
	                    
        chrg = np.sum(wsp.getw())
        erg = np.sum(0.5 / np.abs(charge / mass) * wsp.getw()[0:N_mp] * (wsp.getvx()[0:N_mp] * wsp.getvx()[0:N_mp] + wsp.getvy()[0:N_mp] * wsp.getvy()[0:N_mp] + wsp.getvz()[0:N_mp] * wsp.getvz()[0:N_mp]))        
        print 'Done SOFT regeneration. N_mp=%d Nel_tot=%1.2e En_tot=%1.2e'%(N_mp, chrg, erg)

