def perform_regeneration(target_MP, x_mp,y_mp,z_mp,vx_mp,vy_mp,vz_mp,nel_mp,N_mp):
    return x_mp,y_mp,z_mp,vx_mp,vy_mp,vz_mp,nel_mp,N_mp

def warp_pyecloud_dipole(z_length = 1., nx = 16, ny = 16, nz = 30, n_bunches = 5,
                         b_spac = 25e-9, beam_gamma = 479., sigmax = 2e-4,
                         sigmay = 2.1e-4, sigmat= 1.000000e-09/4.,
                         bunch_intensity = 1e11, init_num_elecs = 1.e5,
                         init_num_elecs_mp = 10**2, By = 0.53,
                         pyecloud_nel_mp_ref = 1., dt = 25e-12,
                         pyecloud_fact_clean = 1e-6, pyecloud_fact_split = 1.5,
                         chamber_type = 'rect', flag_save_video = False,
                         enable_trap = True, Emax = 300., del_max = 1.8,
                         R0 = 0.7, E_th = 30, sigmafit = 1.09, mufit = 1.66,
                         secondary_angle_distribution = 'cosine_3D'):
    
    import numpy as np
    import numpy.random as random
    from warp import picmi
    from scipy.stats import gaussian_kde
    from warp.particles.Secondaries import Secondaries, top, warp, time
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from io import BytesIO as StringIO
    from mpi4py import MPI
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
    bunch_w = 1e6
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
    temp_file_name = 'temp_mps_info.h5'
    
    N_mp_max = 10
    

    if not os.path.exists(temp_file_name):
        x0 = random.uniform(lower_bound[0],upper_bound[0],init_num_elecs_mp)
        y0 = random.uniform(lower_bound[1],upper_bound[1],init_num_elecs_mp)
        z0 = random.uniform(lower_bound[2],upper_bound[2],init_num_elecs_mp)
        vx0 = np.zeros(init_num_elecs_mp)
        vy0 = np.zeros(init_num_elecs_mp)
        vz0 = np.zeros(init_num_elecs_mp)

        w0 = float(init_num_elecs)/float(init_num_elecs_mp)*np.ones(init_num_elecs_mp)
        b_pass_prev = 0
    else:
        print('############################################################')
        print('Temp distribution found. Regenarating and restarting')
        print('############################################################')
        dict_init_dist = dict_of_arrays_and_scalar_from_h5(temp_file_name)
        xold = dict_init_dist['x_mp']
        yold = dict_init_dist['y_mp']
        zold = dict_init_dist['z_mp']
        uxold = dict_init_dist['ux_mp']
        uyold = dict_init_dist['uy_mp']
        uzold = dict_init_dist['uz_mp']
        wold = dict_init_dist['w_mp']
        numelecs = dict_init_dist['numelecs']
        elecs_density = dict_init_dist['elecs_density']
        N_mp = dict_init_dist['N_mp']
        b_pass_prev = dict_init_dist['b_pass']

        invgamma = np.sqrt(1-picmi.clight**2/(np.square(uxold)+np.square(uyold)+np.square(uzold)))
        vxold = np.multiply(invgamma,uxold)
        vyold = np.multiply(invgamma,uxold)
        vzold = np.multiply(invgamma,uxold)

        x0,y0,z0,vx0,vy0,vz0,w0,N_mp = perform_regeneration(xold,yold,zold,vxold,vyold,vzold,wold)



    electron_background_dist = picmi.ParticleListDistribution(x0=x0, y0=y0,
                                                              z0=z0, vx0=vx0,
                                                              vy0=vy0, vz0=vz0,
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
        if l_force or 0==0:#pw.top.it%1==0:
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
            print('plot')

    if flag_save_video:
        pw.installafterstep(myplots)
        myplots(1)

    ntsteps_p_bunch = b_spac/top.dt
    tot_nsteps = int(np.ceil(b_spac*n_bunches/top.dt))

    # pre-allocate outputs
    if not os.path.exists('temp_mps_info.h5'):
        numelecs = np.zeros(tot_nsteps)
        elecs_density = np.zeros((tot_nsteps,nx+1,ny+1,3))
        beam_density = np.zeros((tot_nsteps,nx+1,ny+1,3))
        N_mp = np.zeros(tot_nsteps)

    dict_out = {}

    # aux variables
    b_pass = 0
    perc = 10
    original = sys.stdout
    text_trap = StringIO()
    t0 = time.time()

    # trapping warp std output
    enable_trap = False
    text_trap = {True: StringIO(), False: sys.stdout}[enable_trap]
    original = sys.stdout

    for n_step in range(tot_nsteps):
        # if a passage is starting...
        if n_step/ntsteps_p_bunch > b_pass:
            b_pass+=1
            perc = 10
            print('===========================')
            print('Bunch passage: %d' %(b_pass+b_pass_prev))
            print('Number of electrons in the dipole: %d' %(np.sum(secelec.wspecies.getw())+np.sum(elecb.wspecies.getw())))
            if secelec.wspecies.getn() > N_mp_max:
                dict_out_temp = {}
                print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                print('Please restart the simulation')
                dict_out_temp['x_mp'] = secelec.wspecies.getx()
                dict_out_temp['y_mp'] = secelec.wspecies.gety()
                dict_out_temp['z_mp'] = secelec.wspecies.getz()
                dict_out_temp['ux_mp'] = secelec.wspecies.getux()
                dict_out_temp['uy_mp'] = secelec.wspecies.getuy()
                dict_out_temp['uz_mp'] = secelec.wspecies.getuz()
                dict_out_temp['w_mp'] = secelec_w
                
                dict_out_temp['numelecs'] = numelecs
                dict_out_temp['elecs_density'] = elecs_density
                dict_out_temp['N_mp'] = N_mp
                
                dict_out_temp['b_pass'] = b_pass
                
                filename = 'temp_mps_info.h5'

                mfm.dict_to_h5(dict_out_temp, filename)

        if n_step%ntsteps_p_bunch/ntsteps_p_bunch*100>=perc:
            print('%d%% of bunch passage' %perc)
            perc = perc+10
        
        sys.stdout = text_trap
        step(1)
        sys.stdout = original

        # store outputs
        secelec_w = secelec.wspecies.getw()
        elecb_w = elecb.wspecies.getw()
        numelecs[n_step] = np.sum(secelec_w)+np.sum(elecb_w)
        elecs_density[n_step,:,:,:] = secelec.wspecies.get_density()[:,:,(nz+1)/2-1:(nz+1)/2+2] + elecb.wspecies.get_density()[:,:,(nz+1)/2-1:(nz+1)/2+2]
        N_mp[n_step] = len(secelec_w)+len(elecb_w)


    t1 = time.time()
    totalt = t1-t0
    dict_out['numelecs'] = numelecs
    dict_out['elecs_density'] = elecs_density
    dict_out['N_mp'] = N_mp

    mfm.dict_to_h5(dict_out, 'output.h5')

    if os.path.exists('temp_mps_info.h5'):
        os.remove('temp_mps_info.h5')
    
    print('Run terminated in %ds' %totalt)


