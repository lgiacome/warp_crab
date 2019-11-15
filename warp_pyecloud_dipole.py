from perform_regeneration import perform_regeneration

def warp_pyecloud_dipole(z_length = None, nx = None, ny = None, nz =None, dh_t = None,
                         n_bunches = None, b_spac = None, beam_gamma = None,
                         sigmax = None, sigmay = None, sigmat = None,
                         bunch_intensity = None, init_num_elecs = None,
                         init_num_elecs_mp = None, By = None,
                         pyecloud_nel_mp_ref = None, dt = None,
                         pyecloud_fact_clean = None, pyecloud_fact_split = None,
                         chamber_type = None, flag_save_video = None,
                         enable_trap = True, Emax = None, del_max = None,
                         R0 = None, E_th = None, sigmafit = None, mufit = None,
                         secondary_angle_distribution = None, N_mp_max = None,
                 N_mp_target = None, flag_checkpointing = False, checkpoints = None,
             flag_output = False, bunch_macro_particles = None, t_offs = None,
             width = None, height = None, output_name = 'output.mat', flag_relativ_tracking = False,
             nbins = 100, r=None):
    
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
    
    # Consistency check
    if chamber_type == 'rect':
        assert (height is not None) and (width is not None), 'When using rectangular chamber, height and width must be specified'
    if chamber_type == 'circle':
        assert r is not None, 'When using circular chamber r must be specified'
    if chamber_type == 'LHC':
        h = 18.e-3

    l = z_length
    
    unit = 1e-3
    ghost = 1e-3

    if chamber_type == 'rect':
        xmin = -width/2. - ghost
        xmax = -xmin
        ymin = -height/2 - ghost
        ymax = -ymin
    
        lower_bound = [-width/2,-height/2,zs_dipo]
        upper_bound = [width/2,height/2,ze_dipo]
    else:
        xmin = -r - ghost
        xmax = r + ghost
        ymin = -r - ghost
        ymax = r+ghost
    
        lower_bound = [-r,-r,zs_dipo]
        upper_bound = [r,r,ze_dipo]

    zmin = zs_dipo - 50*unit
    zmax = ze_dipo + 50*unit
    
    if chamber_type == 'LHC':
        chamber_area = 0.0014664200235342726
    elif chamber_type == 'rect':
        chamber_area = width*height
    elif chamber_type == 'circle':
        chamber_area = np.pi*r*r

    init_dist_area = (upper_bound[0]-lower_bound[0])*(upper_bound[1]-lower_bound[1])
    ##########################
    # Beam parameters
    ##########################
    sigmaz = sigmat*picmi.clight
    bunch_w = bunch_intensity/bunch_macro_particles
    print('DEBUG: bunch_w: %d' %bunch_w)
    print('DEBUG: bunch_intensity: %d' %bunch_intensity)
    print('DEBUG: bunch_macro_particles: %d' %bunch_macro_particles)

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
    temp_file_name = 'temp_mps_info.mat'
    
    ########################################################
    # if checkopoint is found reload it, 
    # otherwise start from scratch
    ########################################################
    if flag_checkpointing and os.path.exists(temp_file_name):
        print('############################################################')
        print('Temp distribution found. Regenarating and restarting')
        print('############################################################')
        dict_init_dist = sio.loadmat(temp_file_name)
        # Load particles status
        x0 = dict_init_dist['x_mp'][0]
        y0 = dict_init_dist['y_mp'][0]
        z0 = dict_init_dist['z_mp'][0]
        ux0 = dict_init_dist['ux_mp'][0]
        uy0 = dict_init_dist['uy_mp'][0]
        uz0 = dict_init_dist['uz_mp'][0]
        w0 = dict_init_dist['w_mp'][0]
        # compute the velocities
        invgamma = np.sqrt(1-picmi.clight**2/(np.square(ux0)+np.square(uy0)+np.square(uz0)))
        vx0 = np.multiply(invgamma,ux0)
        vy0 = np.multiply(invgamma,uy0)
        vz0 = np.multiply(invgamma,uy0)
        # Reload the outputs and other auxiliary stuff
        if flag_output:
            numelecs = dict_init_dist['numelecs'][0]
            N_mp = dict_init_dist['N_mp'][0]
            numelecs_tot = dict_init_dist['numelecs_tot'][0]
            xhist = dict_init_dist['xhist'][0]
            bins = dict_init_dist['bins'][0]
            b_pass_prev = dict_init_dist['b_pass'] -1
    else:
        x0 = random.uniform(lower_bound[0],upper_bound[0],init_num_elecs_mp)
        y0 = random.uniform(lower_bound[1],upper_bound[1],init_num_elecs_mp)
        z0 = random.uniform(lower_bound[2],upper_bound[2],init_num_elecs_mp)
        vx0 = np.zeros(init_num_elecs_mp)
        vy0 = np.zeros(init_num_elecs_mp)
        vz0 = np.zeros(init_num_elecs_mp)
        w0 = float(init_num_elecs)/float(init_num_elecs_mp)
        #Correct the weight to the chambers size
        w0 = w0*init_dist_area/chamber_area
        b_pass_prev = 0

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

    if chamber_type == 'rect':
        print 'Using rectangular chamber with xaper: %1.2e, yaper: %1.2e' %(width/2., height/2.)
        upper_box = picmi.warp.YPlane(y0=height/2.,ysign=1,condid=1)
        lower_box = picmi.warp.YPlane(y0=-height/2.,ysign=-1,condid=1)
        left_box = picmi.warp.XPlane(x0=width/2.,xsign=1,condid=1)
        right_box = picmi.warp.XPlane(x0=-width/2.,xsign=-1,condid=1)

        sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                           warp_initialize_solver_after_generate = 1)

        sim.conductors = upper_box + lower_box + left_box + right_box

    elif chamber_type == 'LHC':
        print 'Using the LHC chamber'
        pipe_annulus = picmi.warp.ZAnnulus(rmin = r, rmax    = r+ghost/2, length  = l, condid =1)
        upper_box = picmi.warp.YPlane(y0=h,ysign=1,condid=1)
        lower_box = picmi.warp.YPlane(y0=-h,ysign=-1,condid=1)
        pipe = picmi.warp.ZCylinderOut(radius = r, length = l,condid = 1)

        sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                               warp_initialize_solver_after_generate = 1)

        sim.conductors = pipe+upper_box+lower_box

    elif chamber_type == 'circle':
        print 'Using a circular chamber with radius %d' %r
        pipe_annulus = picmi.warp.ZAnnulus(rmin = r, rmax    = r+ghost, length  = l,
                            voltage = 0., xcent   = 0., ycent   = 0.,
                            zcent   = 0., condid  = 1)
        pipe = picmi.warp.ZCylinderOut(radius = r, length = l,condid = 1) 

        sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                               warp_initialize_solver_after_generate = 1)

        sim.conductors = pipe

    beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

    sim.add_species(beam, layout=None,
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
        for i in range(0,n_bunches):
            val += bunch_macro_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac-t_offs)*(t-i*b_spac-t_offs)/(2*sigmat*sigmat))*picmi.warp.top.dt
        return val

    def nonlinearsource():
        NP = int(round(time_prof(top.time)))
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

    #Subcycle(10)
    
    # just some shortcuts
    pw = picmi.warp
    step = pw.step

    if mysolver=='ES':
        pw.top.dt = dt
        if flag_relativ_tracking:
            pw.top.lrelativ = pw.true
        else:
            pw.top.lrelativ = pw.false

    def myplots(l_force=0):
        if l_force or pw.top.it%10==0:
            plt.close()
            (Nx,Ny,Nz) = np.shape(secelec.wspecies.get_density())
            fig, axs = plt.subplots(1, 2,figsize=(12, 4.5))
            fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
            d = secelec.wspecies.get_density() + elecb.wspecies.get_density() + beam.wspecies.get_density()
            d2 = secelec.wspecies.get_density()+ elecb.wspecies.get_density()
            im1 = axs[0].imshow(d2[:,:,Nz/2] .T,cmap='jet',origin='lower', vmin=0.2*np.min(d2[:,:,Nz/2]), vmax=0.8*np.max(d2[:,:,Nz/2]),extent=[xmin, xmax , ymin, ymax])
            axs[0].set_xlabel('x [m]')
            axs[0].set_ylabel('y [m]')
            axs[0].set_title('e- density')
            fig.colorbar(im1, ax=axs[0],)
            im2 = axs[1].imshow(d2[Nx/2,:,:],cmap='jet',origin='lower', vmin=0.2*np.min(d2[Nx/2,:,:]), vmax=0.8*np.max(d2[Nx/2,:,:]),extent=[zmin, zmax , ymin, ymax], aspect = 'auto')
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
    if flag_output and not (flag_checkpointing and os.path.exists('temp_mps_info.mat')):
        numelecs = np.zeros(tot_nsteps)
        numelecs_tot = np.zeros(tot_nsteps)
        N_mp = np.zeros(tot_nsteps)
        xhist = np.zeros((n_bunches,nbins))

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
            # Measure the duration of the previous passage
            if b_pass>1:
                t_pass_1 = time.time()
                t_pass = t_pass_1-t_pass_0
            t_pass_0 = time.time()
            # Dump outputs
            if flag_output:
                dict_out['numelecs'] = numelecs
                dict_out['numelecs_tot'] = numelecs_tot
                dict_out['N_mp'] = N_mp
                # Compute the x-position histogram
                (xhist[b_pass-1], bins) = np.histogram(secelec.wspecies.getx(), range = (xmin,xmax), bins = nbins, weights = secelec.wspecies.getw(), density = False) 
                dict_out['bins'] = bins
                dict_out['xhist'] = xhist
                sio.savemat(output_name, dict_out)
            # Perform regeneration if needed
            if secelec.wspecies.getn() > N_mp_max:
                dict_out_temp = {}
                print('Number of macroparticles: %d' %(secelec.wspecies.getn()))
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
                        dict_out_temp['numelecs_tot'] = numelecs_tot
                        dict_out_temp['N_mp'] = N_mp
                        dict_out_temp['xhist'] = xhist
                        dict_out_temp['bins'] = bins

                    dict_out_temp['b_pass'] = b_pass + b_pass_prev

                    filename = 'temp_mps_info.mat'

                    sio.savemat(filename, dict_out_temp)

            print('===========================')
            print('Bunch passage: %d' %(b_pass+b_pass_prev))
            print('Number of electrons in the dipole: %d' %(np.sum(secelec.wspecies.getw())+np.sum(elecb.wspecies.getw())))
            print('Number of macroparticles: %d' %(secelec.wspecies.getn()+elecb.wspecies.getn()))
            if b_pass > 1:
                print('Previous passage took %ds' %t_pass)
        if n_step%ntsteps_p_bunch/ntsteps_p_bunch*100>perc:
            print('%d%% of bunch passage' %perc)
            perc = perc+10

        # Perform a step
        sys.stdout = text_trap
        step(1)
        sys.stdout = original
        #if secelec.wspecies.getn()>0 and elecb.wspecies.getn()>0:
        #    print(max(max(np.sqrt(np.square(elecb.wspecies.getvx())+np.square(elecb.wspecies.getvy())+np.square(elecb.wspecies.getvz()))), max(np.sqrt(np.square(secelec.wspecies.getvx())+np.square(secelec.wspecies.getvy())+np.square(secelec.wspecies.getvz())))))
        # Store stuff to be saved
        if flag_output:
            secelec_w = secelec.wspecies.getw()
            elecb_w = elecb.wspecies.getw()
            elecs_density = secelec.wspecies.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)] + elecb.wspecies.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)]
            numelecs[n_step] = np.sum(elecs_density)
            elecs_density_tot = secelec.wspecies.get_density(l_dividebyvolume=0)[:,:,:] + elecb.wspecies.get_density(l_dividebyvolume=0)[:,:,:]
            numelecs_tot[n_step] = np.sum(elecs_density_tot)
            N_mp[n_step] = len(secelec_w)+len(elecb_w)

    # Timer
    t1 = time.time()
    totalt = t1-t0
    # Dump outputs
    if flag_output:
        dict_out['numelecs'] = numelecs
        dict_out['N_mp'] = N_mp
        dict_out['numelecs_tot'] = numelecs_tot
        dict_out['xhist'] = xhist
        dict_out['bins'] = bins
        sio.savemat(output_name, dict_out)
    

    # Delete checkpoint if found
    if flag_checkpointing and os.path.exists('temp_mps_info.mat'):
        os.remove('temp_mps_info.mat')

    print('Run terminated in %ds' %totalt)

