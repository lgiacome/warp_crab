from perform_regeneration import perform_regeneration
import numpy as np
import numpy.random as random
from warp import picmi
from scipy.stats import gaussian_kde
from warp.particles.Secondaries import Secondaries, top, warp, time
import matplotlib.pyplot as plt
from io import StringIO
from scipy.constants import c as clight
import sys
import PyECLOUD.myfilemanager as mfm
import scipy.io as sio
import os
import PyECLOUD.sec_emission_model_ECLOUD as seec
from saver import Saver

class warp_pyecloud_sim:

    def __init__(self, z_length = None, nx = None, ny = None, nz =None, 
                 solver_type = 'ES', n_bunches = None, b_spac = None, 
                 beam_gamma = None, sigmax = None, sigmay = None, 
                 sigmat = None, bunch_intensity = None, init_num_elecs = None,
                 init_num_elecs_mp = None, By = None, N_subcycle = None,
                 pyecloud_nel_mp_ref = None, dt = None, 
                 pyecloud_fact_clean = None, pyecloud_fact_split = None,
                 chamber_type = None, flag_save_video = None,
                 enable_trap = True, Emax = None, del_max = None, R0 = None, 
                 E_th = None, sigmafit = None, mufit = None,
                 secondary_angle_distribution = None, N_mp_max = None,
                 N_mp_target = None, flag_checkpointing = False, 
                 checkpoints = None, flag_output = False, 
                 bunch_macro_particles = None, t_offs = None, width = None, 
                 height = None, output_filename = 'output.mat', 
                 flag_relativ_tracking = False, nbins = 100, radius = None, 
                 ghost = None,ghost_z = None, stride_imgs = 10, 
                 stride_output = 1000,chamber = False, lattice_elem = None, 
                 temps_filename = 'temp_mps_info.mat', custom_plot = None):
        

        # Construct PyECLOUD secondary emission object
        sey_mod = seec.SEY_model_ECLOUD(Emax = Emax, del_max = del_max, R0 = R0,
                                       E_th = E_th, sigmafit = sigmafit,
                                       mufit = mufit,
                                       secondary_angle_distribution='cosine_3D')

        self.nbins = nbins
        self.N_mp_target = N_mp_target
        self.N_mp_max = N_mp_max
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.flag_checkpointing = flag_checkpointing
        self.checkpoints = checkpoints 
        self.flag_output = flag_output
        self.output_filename = output_filename 
        self.stride_imgs = stride_imgs
        self.stride_output = stride_output
        self.beam_gamma = beam_gamma
        self.chamber = chamber
        self.init_num_elecs_mp = init_num_elecs_mp
        self.init_num_elecs = init_num_elecs
        self.n_bunches = n_bunches
        self.bunch_macro_particles = bunch_macro_particles
        self.sigmat = sigmat
        self.b_spac = b_spac
        self.t_offs = t_offs
        self.temps_filename = temps_filename
        # Just some shortcuts
        pw = picmi.warp
        step = pw.step

        if solver_type == 'ES':
            pw.top.dt = dt
            if flag_relativ_tracking:
                pw.top.lrelativ = pw.true
            else:
                pw.top.lrelativ = pw.false


        self.tot_nsteps = int(np.ceil(b_spac*(n_bunches)/top.dt))

        self.saver = Saver(flag_output, flag_checkpointing, 
                      self.tot_nsteps, n_bunches, nbins, 
                      temps_filename = temps_filename,
                      output_filename = output_filename)
 

        # Beam parameters
        sigmaz = sigmat*picmi.clight
        if bunch_macro_particles > 0:
            self.bunch_w = bunch_intensity/bunch_macro_particles
        else:
            self.bunch_w = 0

        self.bunch_rms_size = [sigmax, sigmay, sigmaz]
        self.bunch_rms_velocity = [0., 0., 0.]
        self.bunch_centroid_position = [0, 0, chamber.zmin + 10e-3]
        self.bunch_centroid_velocity = [0.,0., beam_gamma*picmi.constants.c]

        # Instantiate beam
        self.beam = picmi.Species(particle_type = 'proton',
                             particle_shape = 'linear',
                             name = 'beam')
        
        # If checkopoint is found reload it, 
        # otherwise start with uniform distribution
        self.temp_file_name = 'temp_mps_info.mat'
        if self.flag_checkpointing and os.path.exists(self.temp_file_name): 
            electron_background_dist = self.load_elec_density()
        else:
            electron_background_dist = self.init_uniform_density() 

        self.elecb = picmi.Species(particle_type = 'electron',
                              particle_shape = 'linear',
                              name = 'Electron background',
                              initial_distribution = electron_background_dist)

        self.secelec = picmi.Species(particle_type = 'electron',
                                particle_shape = 'linear',
                                name = 'Secondary electrons')

        # Setup grid and boundary conditions
        if solver_type == 'ES':
            lower_bc = ['dirichlet', 'dirichlet', 'dirichlet']
            upper_bc = ['dirichlet', 'dirichlet', 'dirichlet']
        if solver_type == 'EM':
            lower_boundary_conditions = ['open', 'open', 'open']
            upper_boundary_conditions = ['open', 'open', 'open']

        grid = picmi.Cartesian3DGrid(number_of_cells = [self.nx,
                                                        self.ny, 
                                                        self.nz],
                                     lower_bound = [chamber.xmin, 
                                                    chamber.ymin, 
                                                    chamber.zmin],
                                     upper_bound = [chamber.xmax, 
                                                    chamber.ymax, 
                                                    chamber.zmax],
                                     lower_boundary_conditions = lower_bc,
                                     upper_boundary_conditions = upper_bc)

        if solver_type == 'ES':
            solver = picmi.ElectrostaticSolver(grid = grid)
        elif solver_type == 'EM':
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

        # Setup simulation
        sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                               warp_initialize_solver_after_generate = 1)

        sim.conductors = chamber.conductors

        sim.add_species(self.beam, layout = None,
                        initialize_self_field = solver == 'EM')

        self.elecb_layout = picmi.PseudoRandomLayout(
                                          n_macroparticles = init_num_elecs_mp,
                                          seed = 3)

        sim.add_species(self.elecb, layout = self.elecb_layout,
                        initialize_self_field = solver == 'EM')

        sim.add_species(self.secelec, layout = None, 
                        initialize_self_field = False)

        picmi.warp.installuserinjection(self.bunched_beam)

        sim.step(1)
        solver.solver.installconductor(sim.conductors, 
                                       dfill = picmi.warp.largepos)
        sim.step(1)

        # Setup secondary emission stuff       
        pp = warp.ParticleScraper(sim.conductors, lsavecondid = 1, 
                                  lsaveintercept = 1,lcollectlpdata = 1)

        sec=Secondaries(conductors = sim.conductors, l_usenew = 1,
                        pyecloud_secemi_object = sey_mod,
                        pyecloud_nel_mp_ref = pyecloud_nel_mp_ref,
                        pyecloud_fact_clean = pyecloud_fact_clean,
                        pyecloud_fact_split = pyecloud_fact_split)

        sec.add(incident_species = self.elecb.wspecies,
                emitted_species  = self.secelec.wspecies,
                conductor        = sim.conductors)

        sec.add(incident_species = self.secelec.wspecies,
                emitted_species = self.secelec.wspecies,
                conductor       = sim.conductors)

        if N_subcycle is not None:
            Subcycle(N_subcycle)
        
        if custom_plot is not None:
            plot_func = custom_plot
        else:
            plot_func = self.myplots

        pw.installafterstep(plot_func)
        plot_func(1)

        self.ntsteps_p_bunch = b_spac/top.dt
        t_start = self.b_pass_prev*b_spac
        tstep_start = int(np.round(t_start/top.dt))

        # aux variables
        self.b_pass = 0
        self.perc = 10
        self.t0 = time.time()
        
        # trapping warp std output
        self.text_trap = {True: StringIO(), False: sys.stdout}[enable_trap]
        self.original = sys.stdout

        self.n_step = int(np.round(self.b_pass_prev*b_spac/dt))
         
        #breakpoint()

    def step(self, u_steps = 1):
        for u_step in range(u_steps):
            # if a passage is starting...
            if (self.n_step/self.ntsteps_p_bunch 
                    >= self.b_pass + self.b_pass_prev):
                self.b_pass+=1
                self.perc = 10
                # Measure the duration of the previous passage
                if self.b_pass>1:
                    self.t_pass_1 = time.time()
                    self.t_pass = self.t_pass_1-self.t_pass_0
                self.t_pass_0 = time.time()
                # Perform regeneration if needed
                if self.secelec.wspecies.getn() > self.N_mp_max:
                    print('Number of macroparticles: %d' 
                          %(self.secelec.wspecies.getn()))
                    print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    perform_regeneration(self.N_mp_target, 
                                         self.secelec.wspecies, sec)
                 
                # Save stuff if checkpoint
                if (self.flag_checkpointing 
                   and np.any(self.checkpoints == self.b_pass + self.b_pass_prev)):
                    self.saver.save_checkpoint(self.b_pass + self.b_pass_prev, 
                                               self.elecb.wspecies, 
                                               self.secelec.wspecies)

                print('===========================')
                print('Bunch passage: %d' %(self.b_pass+self.b_pass_prev))
                print('Number of electrons in the dipole: %d' 
                        %(np.sum(self.secelec.wspecies.getw())
                          + np.sum(self.elecb.wspecies.getw())))
                print('Number of macroparticles: %d' 
                        %(self.secelec.wspecies.getn() 
                          + self.elecb.wspecies.getn()))
                if self.b_pass > 1:
                    print('Previous passage took %ds' %self.t_pass)


            if (self.n_step%self.ntsteps_p_bunch/self.ntsteps_p_bunch*100 
                        > self.perc):
                print('%d%% of bunch passage' %self.perc)
                self.perc = self.perc + 10

            # Dump outputs
            if self.flag_output and self.n_step%self.stride_output == 0:
                self.saver.dump_outputs(self.chamber.xmin, self.chamber.xmax, 
                                        self.secelec.wspecies, self.b_pass)

            # Perform a step
            sys.stdout = self.text_trap
            picmi.warp.step(1)
            sys.stdout = self.original
            #print(sum(self.secelec.wspecies.getw())+sum(self.elecb.wspecies.getw()))

            # Store stuff to be saved
            if self.flag_output:
                self.saver.update_outputs(self.secelec.wspecies, 
                                          self.elecb.wspecies, self.nz,
                                          self.n_step) 
            self.n_step += 1

            if self.n_step > self.tot_nsteps:
                # Timer
                t1 = time.time()
                totalt = t1-self.t0
                # Delete checkpoint if found
                if flag_checkpointing and os.path.exists('temp_mps_info.mat'):
                    os.remove('temp_mps_info.mat')

                print('Run terminated in %ds' %totalt)

    def all_steps(self):
        self.step(self.tot_nsteps)

    def init_uniform_density(self):
        chamber = self.chamber
        lower_bound = chamber.lower_bound
        upper_bound = chamber.upper_bound
        init_num_elecs_mp = self.init_num_elecs_mp
        x0 = random.uniform(lower_bound[0], upper_bound[0],
                            init_num_elecs_mp)
        y0 = random.uniform(lower_bound[1], upper_bound[1],
                            init_num_elecs_mp)
        z0 = random.uniform(lower_bound[2], upper_bound[2],
                            init_num_elecs_mp)
        vx0 = np.zeros(init_num_elecs_mp)
        vy0 = np.zeros(init_num_elecs_mp)
        vz0 = np.zeros(init_num_elecs_mp)

        flag_out = chamber.is_outside(x0, y0, z0)
        Nout = np.sum(flag_out)
        while Nout>0:
            x0[flag_out] = random.uniform(lower_bound[0],upper_bound[0],Nout)
            y0[flag_out] = random.uniform(lower_bound[1],upper_bound[1],Nout)

            flag_out = chamber.isOutside(x0, y0, z0)
            Nout = np.sum(flag_out)
            
        w0 = float(self.init_num_elecs)/float(init_num_elecs_mp)         

    
        self.b_pass_prev = 0

        return picmi.ParticleListDistribution(x = x0, y = y0, z = z0, vx = vx0,
                                              vy = vy0, vz = vz0, weight = w0)



    def load_elec_density(self):
        print('#############################################################')
        print('Temp distribution found. Reloading it as initial distribution')
        print('#############################################################')
        dict_init_dist = sio.loadmat(self.temp_file_name)
        # Load particles status
        x0 = dict_init_dist['x_mp'][0]
        y0 = dict_init_dist['y_mp'][0]
        z0 = dict_init_dist['z_mp'][0]
        vx0 = dict_init_dist['vx_mp'][0]
        vy0 = dict_init_dist['vy_mp'][0]
        vz0 = dict_init_dist['vz_mp'][0]
        w0 = dict_init_dist['nel_mp'][0]
        
        self.b_pass = dict_init_dist['b_pass'][0][0]
        self.b_pass_prev = dict_init_dist['b_pass'][0][0] - 1

    

        return picmi.ParticleListDistribution(x = x0, y = y0, z = z0, vx = vx0,
                                              vy = vy0, vz = vz0, weight = w0)

           

    def time_prof(self, t):
        val = 0
        for i in range(0,self.n_bunches):
            val += (self.bunch_macro_particles*1.
                   /np.sqrt(2*np.pi*self.sigmat**2)
                   *np.exp(-(t-i*self.b_spac-self.t_offs)**2
                   /(2*self.sigmat**2))*picmi.warp.top.dt)
        return val

    def bunched_beam(self):
        NP = int(np.round(self.time_prof(top.time)))
        x = random.normal(self.bunch_centroid_position[0], 
                          self.bunch_rms_size[0], NP)
        y = random.normal(self.bunch_centroid_position[1], 
                          self.bunch_rms_size[1], NP)
        z = self.bunch_centroid_position[2]
        vx = random.normal(self.bunch_centroid_velocity[0], 
                           self.bunch_rms_velocity[0], NP)
        vy = random.normal(self.bunch_centroid_velocity[1], 
                           self.bunch_rms_velocity[1], NP)
        vz = picmi.warp.clight*np.sqrt(1 - 1./(self.beam_gamma**2))
        self.beam.wspecies.addparticles(x = x, y = y, z = z, vx = vx, vy = vy, 
                                        vz = vz, gi = 1./self.beam_gamma,
                                        w = self.bunch_w)

    def myplots(self, l_force=0):
        chamber = self.chamber
        if l_force or self.n_step%self.stride_imgs == 0:
            plt.close()
            (Nx, Ny, Nz) = np.shape(self.secelec.wspecies.get_density())
            fig, axs = plt.subplots(1, 2, figsize = (12, 4.5))
            fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                                top = 0.94, wspace = 0.15)
            d = (self.secelec.wspecies.get_density()
               + self.elecb.wspecies.get_density()
               + self.beam.wspecies.get_density())
            d2  = (self.secelec.wspecies.get_density() 
                + self.elecb.wspecies.get_density())
            im1 = axs[0].imshow(d[:, :, int(Nz/2)] .T, cmap = 'jet', 
                  origin = 'lower', vmin = 0.2*np.min(d2[:, :, int(Nz/2)]), 
                  vmax = 0.8*np.max(d2[:, :, int(Nz/2)]), 
                  extent = [chamber.xmin, chamber.xmax , 
                            chamber.ymin, chamber.ymax])
            axs[0].set_xlabel('x [m]')
            axs[0].set_ylabel('y [m]')
            axs[0].set_title('e- density')
            fig.colorbar(im1, ax = axs[0])
            im2 = axs[1].imshow(d[int(Nx/2), :, :], cmap = 'jet', 
                                origin = 'lower', 
                                vmin = 0.2*np.min(d2[int(Nx/2), :, :]), 
                                vmax = 0.8*np.max(d2[int(Nx/2), :, :]),
                                extent=[chamber.zmin, chamber.zmax, 
                                        chamber.ymin, chamber.ymax], 
                                aspect = 'auto')
            axs[1].set_xlabel('z [m]')
            axs[1].set_ylabel('y [m]')
            axs[1].set_title('e- density')
            fig.colorbar(im2, ax = axs[1])
            n_step = top.time/top.dt
            figname = 'images/%d.png' %n_step
            plt.savefig(figname)


