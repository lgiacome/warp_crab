from perform_regeneration import perform_regeneration

def warp_pyecloud_dipole(z_length = None, nx = None, ny = None, nz =None, n_bunches = None,
                         b_spac = None, beam_gamma = None, sigmax = None,
                         sigmay = None, sigmat = None,
                         bunch_intensity = None, init_num_elecs = None,
                         init_num_elecs_mp = None, By = None,
                         pyecloud_nel_mp_ref = None, dt = None,
                         pyecloud_fact_clean = None, pyecloud_fact_split = None,
                         chamber_type = None, flag_save_video = None,
                         enable_trap = True, Emax = None, del_max = None,
                         R0 = None, E_th = None, sigmafit = None, mufit = None,
                         secondary_angle_distribution = None, N_mp_max = None,
                 N_mp_target = None, flag_checkpointing = False, checkpoints = None,
             flag_output = False):
    
    import numpy as np
    import numpy.random as random
    from warp import picmi
    from scipy.stats import gaussian_kde
    from warp.particles.Secondaries import Secondaries, top, warp, time
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from io import BytesIO as StringIO
    from scipy.constants import c as clight
    import sys
    import PyECLOUD.myfilemanager as mfm
    import os

    # Construct PyECLOUD secondary emission object
    import PyECLOUD.sec_emission_model_ECLOUD as seec
    sey_mod = seec.SEY_model_ECLOUD(Emax = Emax, del_max = del_max, R0 = R0,
                                    E_th = E_th, sigmafit = sigmafit,
                                    mufit = mufit,
                                    secondary_angle_distribution='cosine_3D')

    ##########################
    # physics parameters
    ##########################

    mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

    ##########################
    # numerics parameters
    ##########################

    # --- grid

    zs_dipo = -0.5*z_length
    ze_dipo = 0.5*z_length
    r = 23.e-3
    h = 18.e-3
    l = z_length
    
    unit = 1e-3
    ghost = 1e-3
    xmin = -r-ghost
    xmax = r+ghost
    ymin = -r-ghost
    ymax = r+ghost
    zmin = zs_dipo-50*unit
    zmax = ze_dipo+50*unit
    
    if chamber_type == 'LHC':
        chamber_area = 0.0014664200235342726
    elif chamber_type == 'rect':
        chamber_area = 4*r*h

    ##########################
    # Beam parameters
    ##########################
    sigmaz = sigmat*picmi.clight
    t_offs = b_spac-6*sigmat
    bunch_w = 1e8
    bunch_macro_particles = bunch_intensity/bunch_w
    
    #######################################################
    # compute beam size from normalized emittance and beta
    # Uncomment if data available
    #######################################################
    bunch_rms_size            = [sigmax, sigmay, sigmaz]
    bunch_rms_velocity        = [0.,0.,0.]
    bunch_centroid_position   = [0,0,zs_dipo-10*unit]
    bunch_centroid_velocity   = [0.,0.,beam_gamma*picmi.constants.c]

    beam = picmi.Species(particle_type = 'proton',
                         particle_shape = 'linear',
                         name = 'beam')

    lower_bound = [-r,-h,zs_dipo]
    upper_bound = [r,h,ze_dipo]
    temp_file_name = 'temp_mps_info.mat'
    
    ########################################################
    # if checkopoint is found reload it, 
    # otherwise start from scratch
    ########################################################
    if not os.path.exists(temp_file_name):
        x0 = random.uniform(lower_bound[0],upper_bound[0],init_num_elecs_mp)
        y0 = random.uniform(lower_bound[1],upper_bound[1],init_num_elecs_mp)
        z0 = random.uniform(lower_bound[2],upper_bound[2],init_num_elecs_mp)
        vx0 = np.zeros(init_num_elecs_mp)
        vy0 = np.zeros(init_num_elecs_mp)
    # Initialize with a super small velocity to avoid warning in Secondaries.py
        vz0 = 0.0001*np.ones(init_num_elecs_mp)

        w0 = float(init_num_elecs)/float(init_num_elecs_mp)
        b_pass_prev = 0
    else:
        print('############################################################')
        print('Temp distribution found. Regenarating and restarting')
        print('############################################################')
        dict_init_dist = sio.loadmat(temp_file_name)
    # Load particles status
        x0 = dict_init_dist['x_mp']
        y0 = dict_init_dist['y_mp']
        z0 = dict_init_dist['z_mp']
        ux0 = dict_init_dist['ux_mp']
        uy0 = dict_init_dist['uy_mp']
        uz0 = dict_init_dist['uz_mp']
        w0 = dict_init_dist['w_mp']
    # Reload the outputs and other auxiliary stuff
        numelecs = dict_init_dist['numelecs']
        N_mp = dict_init_dist['N_mp']
        b_pass_prev = dict_init_dist['b_pass'] -1
    # compute the velocities
        invgamma = np.sqrt(1-picmi.clight**2/(np.square(ux0)+np.square(uy0)+np.square(uz0)))
        vx0 = np.multiply(invgamma,ux0)
        vy0 = np.multiply(invgamma,uy0)
        vz0 = np.multiply(invgamma,uy0)

    electron_background_dist = picmi.ParticleListDistribution(x=x0, y=y0,
                                                              z=z0, vx=vx0,
                                                              vy=vy0, vz=vz0,
                                                              weight=w0)

    elecb = picmi.Species(particle_type = 'electron',
                          particle_shape = 'linear',
                          name = 'Electron background',
                          initial_distribution = electron_background_dist)

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
                                 upper_boundary_conditions = upper_boundary_conditions)

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
                                           warp_l_correct_num_Cherenkov = False,
                                           warp_type_rz_depose = 0,
                                           warp_l_setcowancoefs = True,
                                           warp_l_getrho = False)


    ##########################
    # Simulation setup
    ##########################

    pipe = picmi.warp.ZCylinderOut(r,l,condid=1)

    if chamber_type == 'rect':
        upper_box = picmi.warp.YPlane(y0=h,ysign=1,condid=1)
        lower_box = picmi.warp.YPlane(y0=-h,ysign=-1,condid=1)
        left_box = picmi.warp.XPlane(x0=r,xsign=1,condid=1)
        right_box = picmi.warp.XPlane(x0=-r,xsign=-1,condid=1)

        sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                           warp_initialize_solver_after_generate = 1)

        sim.conductors = upper_box + lower_box + left_box + right_box

    elif chamber_type == 'LHC':
        pipe = picmi.warp.ZCylinderOut(r,l,condid=1)
        upper_box = picmi.warp.YPlane(y0=h,ysign=1,condid=1)
        lower_box = picmi.warp.YPlane(y0=-h,ysign=-1,condid=1)

        sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                               warp_initialize_solver_after_generate = 1)

        sim.conductors = pipe+upper_box+lower_box

    beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

    sim.add_species(beam, layout=beam_layout,
                    initialize_self_field = solver=='EM')

    elecb_layout = picmi.PseudoRandomLayout(n_macroparticles=init_num_elecs_mp,
                                            seed = 3)

    sim.add_species(elecb, layout=elecb_layout,
                    initialize_self_field = solver=='EM')
    sim.add_species(secelec, layout=None, initialize_self_field=False)
    #########################
    # Add Dipole
    #########################
    picmi.warp.addnewdipo(zs = zs_dipo, ze = ze_dipo, by = By)


    def time_prof(t):
        val = 0
        sigmat = sigmaz/picmi.clight
        for i in range(1,n_bunches+1):
            val += bunch_macro_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac+t_offs)*(t-i*b_spac+t_offs)/(2*sigmat*sigmat))*picmi.warp.top.dt
        return val

    def nonlinearsource():
        NP = int(time_prof(top.time))
        x = random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
        y = random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
        z = bunch_centroid_position[2]
        vx = random.normal(bunch_centroid_velocity[0],bunch_rms_velocity[0],NP)
        vy = random.normal(bunch_centroid_velocity[1],bunch_rms_velocity[1],NP)
        vz = picmi.warp.clight*np.sqrt(1-1./(beam_gamma**2))
        beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_gamma, w=bunch_w)

    picmi.warp.installuserinjection(nonlinearsource)


    ##########################
    # simulation run
    ##########################
    sim.step(1)
    solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
    sim.step(1)
   
    pp = warp.ParticleScraper(sim.conductors,lsavecondid=1,lsaveintercept=1,lcollectlpdata=1)

    sec=Secondaries(conductors = sim.conductors, l_usenew = 1,
                    pyecloud_secemi_object = sey_mod,
                    pyecloud_nel_mp_ref = pyecloud_nel_mp_ref,
                    pyecloud_fact_clean = pyecloud_fact_clean,
                    pyecloud_fact_split = pyecloud_fact_split)

    sec.add(incident_species = elecb.wspecies,
            emitted_species  = secelec.wspecies,
            conductor        = sim.conductors)

    sec.add(incident_species = secelec.wspecies,
            emitted_species = secelec.wspecies,
            conductor       = sim.conductors)

    # just some shortcuts
    pw = picmi.warp
    step = pw.step

    if mysolver=='ES':
        pw.top.dt = dt


    def myplots(l_force=0):
        if l_force or pw.top.it%10==0:
            plt.close()
            (Nx,Ny,Nz) = np.shape(secelec.wspecies.get_density())
            fig, axs = plt.subplots(1, 2,figsize=(12, 4.5))
            fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
            d = secelec.wspecies.get_density() + elecb.wspecies.get_density() + beam.wspecies.get_density()
            d2 = secelec.wspecies.get_density()+ elecb.wspecies.get_density()
            im1 = axs[0].imshow(d[:,:,Nz/2] .T,cmap='jet',origin='lower', vmin=0.2*np.min(d2[:,:,Nz/2]), vmax=0.8*np.max(d2[:,:,Nz/2]),extent=[xmin, xmax , ymin, ymax])
            axs[0].set_xlabel('x [m]')
            axs[0].set_ylabel('y [m]')
            axs[0].set_title('e- density')
            fig.colorbar(im1, ax=axs[0],)
            im2 = axs[1].imshow(d[Nx/2,:,:],cmap='jet',origin='lower', vmin=0.2*np.min(d2[Nx/2,:,:]), vmax=0.8*np.max(d2[Nx/2,:,:]),extent=[zmin, zmax , ymin, ymax], aspect = 'auto')
            axs[1].set_xlabel('z [m]')
            axs[1].set_ylabel('y [m]')
            axs[1].set_title('e- density')
            fig.colorbar(im2, ax=axs[1])
            n_step = top.time/top.dt
            figname = 'images/%d.png' %n_step
            plt.savefig(figname)

    if flag_save_video:
        pw.installafterstep(myplots)
        myplots(1)

    ntsteps_p_bunch = b_spac/top.dt
    tot_nsteps = int(np.ceil(b_spac*(n_bunches)/top.dt))
    t_start = b_pass_prev*b_spac
    tstep_start = int(round(t_start/top.dt))


    # pre-allocate outputs
    if flag_output and not os.path.exists('temp_mps_info.mat'):
        numelecs = np.zeros(tot_nsteps)
        N_mp = np.zeros(tot_nsteps)

    dict_out = {}

    # aux variables
    b_pass = 0
    perc = 10
    original = sys.stdout
    text_trap = StringIO()
    t0 = time.time()

    # trapping warp std output
    text_trap = {True: StringIO(), False: sys.stdout}[enable_trap]
    original = sys.stdout

    for n_step in range(tstep_start,tot_nsteps):
        # if a passage is starting...
        if n_step/ntsteps_p_bunch >= b_pass+b_pass_prev:
            b_pass+=1
            perc = 10
            # Perform regeneration if needed
            if secelec.wspecies.getn() > N_mp_max:
                dict_out_temp = {}
                print('Number of macroparticles: %d' %(secelec.wspecies.getn()+elecb.wspecies.getn()))
                print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    
                perform_regeneration(N_mp_target, secelec.wspecies, sec)
     
            # Save stuff if checkpoint
                if flag_checkpointing and np.any(checkpoints == b_pass + b_pass_prev) and b_pass>1:
                    dict_out_temp = {}
                    print('Saving a checkpoint!')
                    secelec_w = secelec.wspecies.getw()
                    dict_out_temp['x_mp'] = np.concatenate((secelec.wspecies.getx(),elecb.wspecies.getx()))
                    dict_out_temp['y_mp'] = np.concatenate((secelec.wspecies.gety(),elecb.wspecies.gety()))
                    dict_out_temp['z_mp'] = np.concatenate((secelec.wspecies.getz(),elecb.wspecies.gety()))
                    dict_out_temp['ux_mp'] = np.concatenate((secelec.wspecies.getux(),elecb.wspecies.getux()))
                    dict_out_temp['uy_mp'] = np.concatenate((secelec.wspecies.getuy(),elecb.wspecies.getuy()))
                    dict_out_temp['uz_mp'] = np.concatenate((secelec.wspecies.getuz(),elecb.wspecies.getuz()))
                    dict_out_temp['w_mp'] = np.concatenate((secelec_w,elecb.wspecies.getw()))
                if flag_output:
                    dict_out_temp['numelecs'] = numelecs
                    dict_out_temp['N_mp'] = N_mp

                    dict_out_temp['b_pass'] = b_pass + b_pass_prev

                    filename = 'temp_mps_info.mat'

                if os.path.exists(filename):
                    os.remove(filename)

                sio.savemat(filname, dict_out_temp)

            print('===========================')
            print('Bunch passage: %d' %(b_pass+b_pass_prev))
            print('Number of electrons in the dipole: %d' %(np.sum(secelec.wspecies.getw())+np.sum(elecb.wspecies.getw())))
            print('Number of macroparticles: %d' %(secelec.wspecies.getn()+elecb.wspecies.getn()))
 
        if n_step%ntsteps_p_bunch/ntsteps_p_bunch*100>perc:
            print('%d%% of bunch passage' %perc)
            perc = perc+10

        #if n_step%10==0:
        #    print('Number of macroparticles beam: %d' %(beam.wspecies.getn()))
        #    print('Number of macroparticles: %d' %(secelec.wspecies.getn()+elecb.wspecies.getn()))


    # Perform a step
        sys.stdout = text_trap
        step(1)
        sys.stdout = original

    # Store stuff to be saved
        if flag_output:
            secelec_w = secelec.wspecies.getw()
            elecb_w = elecb.wspecies.getw()
            elecs_density = secelec.wspecies.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)] + elecb.wspecies.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)]
            numelecs[n_step] = np.sum(elecs_density)
            N_mp[n_step] = len(secelec_w)+len(elecb_w)

    # Timer
    t1 = time.time()
    totalt = t1-t0
    # Dump outputs
    if flag_output:
        dict_out['numelecs'] = numelecs
        dict_out['N_mp'] = N_mp

        sio.savemat('output.mat', dict_out)
    

    # Delete checkpoint if found
    if os.path.exists('temp_mps_info.mat'):
        os.remove('temp_mps_info.mat')

    print('Run terminated in %ds' %totalt)

