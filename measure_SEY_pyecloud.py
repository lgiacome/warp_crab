def measure_SEY(Ekin):


    import numpy as np
    import numpy.random as random
    #from pywarpx import picmi
    from warp import picmi
    from scipy.stats import gaussian_kde
    from warp.particles.Secondaries import Secondaries, warp, top, posC, time
    import matplotlib.pyplot as plt
    import parser
    import scipy.io as sio
    from scipy.constants import m_e
    from scipy.constants import c as c_light
    from io import BytesIO as StringIO
    from mpi4py import MPI
    
    # Construct PyECLOUD secondary emission object
    import PyECLOUD.sec_emission_model_ECLOUD as seec
    sey_mod = seec.SEY_model_ECLOUD(Emax=300., del_max=1.8, R0=0.7, E_th=30,
            sigmafit=1.09, mufit=1.66,
            secondary_angle_distribution='cosine_3D')



    ##########################
    # physics parameters
    ##########################

    mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

    # --- Beam

    ##########################
    # numerics parameters
    ##########################
    # geometry
    r = 0.005
    l = 0.1

    # --- grid
    dh = r/10.
    xmin = -2*r + 0.001
    xmax = -xmin
    ymin = -2*r + 0.001
    ymax = -ymin
    zmin = -l - 0.001
    zmax = l

    nx = (xmax-xmin)/dh
    ny = (xmax-xmin)/dh
    nz = (xmax-xmin)/dh

    #######################################################
    # compute beam size from normalized emittance and beta
    # Uncomment if data available
    #######################################################

    sigmax = 2e-4
    sigmay = 2.1e-4
    E0 = 0.511*1e6
    E = E0 + Ekin
    beam_gamma = E/E0
    beam_beta = np.sqrt(1-1/(beam_gamma*beam_gamma))
    v = beam_beta*c_light 

    elec_beam = picmi.Species(particle_type = 'electron',
                         particle_shape = 'linear',
                         name = 'elec_beam')
                         #initial_distribution = gauss_dist)


    secelec = picmi.Species(particle_type = 'electron',
                            particle_shape = 'linear',
                            name = 'Secondary electrons')

    ##########################
    # Numeric components
    ##########################

    if mysolver=='ES':
        lower_boundary_conditions = ['dirichlet', 'dirichlet', 'dirichlet']
        upper_boundary_conditions = ['dirichlet', 'dirichlet', 'dirichlet']
    if mysolver=='EM':
        lower_boundary_conditions = ['open', 'open', 'open']
        upper_boundary_conditions = ['open', 'open', 'open']


    grid = picmi.Cartesian3DGrid(number_of_cells = [nx, ny, nz],
                                 lower_bound = [xmin, ymin, zmin],
                                 upper_bound = [xmax, ymax, zmax],
                                 lower_boundary_conditions = lower_boundary_conditions,
                                 upper_boundary_conditions = upper_boundary_conditions)#,
    #warpx_max_grid_size=32)

    if mysolver=='ES':
        solver = picmi.ElectrostaticSolver(grid = grid)

    if mysolver=='EM':
        smoother = picmi.BinomialSmoother(n_pass = [[1], [1], [1]],
                                          compensation = [[False], [False], [False]],
                                          stride = [[1], [1], [1]],
                                          alpha = [[0.5], [0.5], [0.5]])

        solver = picmi.ElectromagneticSolver(grid = grid,
                                             method = 'CKC',
                                             cfl = 1.,
                                             source_smoother = smoother,
        #                                     field_smoother = smoother,
                                             warp_l_correct_num_Cherenkov = False,
                                             warp_type_rz_depose = 0,
                                             warp_l_setcowancoefs = True,
                                             warp_l_getrho = False)


    ##########################
    # Simulation setup
    ##########################
    wall = picmi.warp.YCylinderOut(r,l)

    sim = picmi.Simulation(solver = solver,
                           verbose = 1,
                           cfl = 1.0,
                           warp_initialize_solver_after_generate = 1)

    sim.conductors = wall

    beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

    sim.add_species(elec_beam, layout=beam_layout,
                    initialize_self_field = solver=='EM')

    sim.add_species(secelec, layout=None, initialize_self_field=False)

    #########################
    # Add Dipole
    #########################
    bunch_w = 1

    def nonlinearsource():
        NP = 1000*(top.it==3)
        x = 0*np.ones(NP)
        y = 0*np.ones(NP)
        z = 0*np.ones(NP)
        vx = np.zeros(NP)
        vy = np.zeros(NP)
        vz = picmi.warp.clight*np.sqrt(1-1./(beam_gamma**2))
        elec_beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_gamma, w=bunch_w)

    picmi.warp.installuserinjection(nonlinearsource)


    ##########################
    # simulation run
    ##########################
    sim.step(1)
    solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
    sim.step(1)


    pp = warp.ParticleScraper(sim.conductors,lsavecondid=1,lsaveintercept=1,lcollectlpdata=1)

    sec=Secondaries(conductors=sim.conductors,
                    l_usenew=1, pyecloud_secemi_object=sey_mod,
                    pyecloud_nel_mp_ref=1., pyecloud_fact_clean=1e-6,
                    pyecloud_fact_split=1.5)

    sec.add(incident_species = elec_beam.wspecies,
            emitted_species  = secelec.wspecies,
            conductor        = sim.conductors)

    # --- set weights of secondary electrons
    #secelec.wspecies.getw()[...] = elecb.wspecies.getw()[0]

    # define shortcuts
    pw = picmi.warp
    em = solver.solver
    step=pw.step
    top.dt = dh/v

    n_step = 0
    tot_t = 2*r/v
    tot_nsteps = int(tot_t/top.dt)
    for n_step in range(tot_nsteps):
        step(1)
    secondaries_count = np.sum(secelec.wspecies.getw())

    return secondaries_count/(100*1000)

import numpy as np

ene_array = np.linspace(200,1001,100)
res = np.zeros_like(ene_array)
from run_in_separate_process import run_in_separate_process
import sys
original = sys.stdout
from io import BytesIO as StringIO
text_trap = StringIO()

for ii, ene in enumerate([5]):
    print(ii)
    sys.stdout = text_trap
    res[ii] = run_in_separate_process(measure_SEY, [ene])
    sys.stdout = original
import matplotlib.pyplot as plt
plt.plot(res)
plt.show()

