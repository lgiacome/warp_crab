import numpy as np
from numpy.random import rand
from warp import echarge as charge
from warp import emass as mass
from warp import top
from warp_parallel import parallelsum

def perform_regeneration(target_N_mp, wsp, Sec):
    # Old number of MPs
    N_mp = wsp.getn()
    # Compute old total charge and Ekin
    chrg = np.sum(wsp.getw())
    erg = np.sum(0.5 / np.abs(charge / mass) * wsp.getw() * (wsp.getvx() * wsp.getvx() + wsp.getvy() * wsp.getvy() + wsp.getvz() * wsp.getvz()))
    
    #Update the reference MP size
    new_nel_mp_ref = chrg / target_N_mp  
    Sec.set_nel_mp_ref(new_nel_mp_ref)

    print 'New nel_mp_ref = %d'%(Sec.pyecloud_nel_mp_ref)
    print 'Start SOFT regeneration. N_mp=%d Nel_tot=%1.2e, En_tot = %1.2e'%(N_mp, chrg, erg)
                
    # Compute the death probability         
    death_prob = float(N_mp - target_N_mp) / float(N_mp)
    # Decide which MPs to keep/discard
    N_mp_local = top.pgroup.nps[wsp.getjs()]
    flag_keep = (rand(N_mp_local) > death_prob)
    
    # Compute the indices of the particles to be kept/cleared       
    i_init = top.pgroup.ins[wsp.getjs()]
    inds_clear = np.where(flag_keep == False)[0] + i_init-1
    inds_keep = np.where(flag_keep == True)[0] + i_init -1

    # Compute the charge after the regeneration in the whole domain (also the other processes)
    chrg_after = parallelsum(np.sum(top.pgroup.pid[inds_keep,top.wpid-1]))
    # Resize the survivors
    correct_fact = chrg / chrg_after
    top.pgroup.pid[inds_keep,top.wpid-1] = top.pgroup.pid[inds_keep,top.wpid-1]*correct_fact
    # Flag the particles to be cleared
    top.pgroup.gaminv[inds_clear] = 0.

    # Warning: top.clearpart is a Fortran routine so it has one-based ordering..
    top.clearpart(top.pgroup, wsp.getjs()+1, 1)        
    # Compute new number of MPs
    N_mp = wsp.getn()

    print 'Applied correction factor = %e'%correct_fact
    chrg = chrg = np.sum(wsp.getw()) 
    erg = np.sum(0.5 / np.abs(charge / mass) * wsp.getw() * (wsp.getvx() * wsp.getvx() + wsp.getvy() * wsp.getvy() + wsp.getvz() * wsp.getvz()))        
    print 'Done SOFT regeneration. N_mp=%d Nel_tot=%1.2e En_tot=%1.2e'%(N_mp, chrg, erg)
