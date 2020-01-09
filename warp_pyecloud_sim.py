from perform_regeneration import perform_regeneration
import numpy as np
import numpy.random as random
from warp import picmi
from scipy.stats import gaussian_kde
from warp.particles.Secondaries import Secondaries, top, warp, time
import matplotlib.pyplot as plt
from io import BytesIO as StringIO
from scipy.constants import c as clight
import sys
import PyECLOUD.myfilemanager as mfm
import scipy.io as sio
import os
import PyECLOUD.sec_emission_model_ECLOUD as seec

class warp_pyecloud_sim:

    def __init__(self, z_length = None, nx = None, ny = None, nz =None, dh_t = None,
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
                 nbins = 100, radius = None, lattice_elem = None, ghost = None,ghost_z = None,
                 stride_imgs = 10, stride_output = 1000):
        

        # Construct PyECLOUD secondary emission object
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
        self.nbins = nbins
        self.N_mp_target = N_mp_target
        self.N_mp_max = N_mp_max
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.flag_checkpointing = flag_checkpointing
        self.flag_output = flag_output
        self.output_name = output_name 
        self.stride_imgs = stride_imgs
        self.stride_output = stride_output
        # --- grid
        if lattice_elem == 'dipole':
            zs_dipo = -0.5*z_length
            ze_dipo = 0.5*z_length
        
        # Consistency check
        if chamber_type == 'rect':
            assert (height is not None) and (width is not None), 'When using rectangular chamber, height and width must be specified'
        if chamber_type == 'circle':
            assert radius is not None, 'When using circular chamber r must be specified'
        if chamber_type == 'LHC':
            radius = 23.e-3
            height = 2*18.e-3
            width = 2*radius
            

        z_length
        
        unit = 1e-3
        if ghost is None:
            ghost = 1e-3

        if ghost_z is None:
            ghost_z = ghost

        if chamber_type == 'rect' or chamber_type == 'LHC':
            self.xmin = -width/2. - ghost
            self.xmax = -self.xmin
            self.ymin = -height/2 - ghost
            self.ymax = -self.ymin
            self.zmin = zs_dipo - 50*unit
            self.zmax = ze_dipo + 50*unit

            lower_bound = [-width/2,-height/2,zs_dipo]
            upper_bound = [width/2,height/2,ze_dipo]
        elif chamber_type == 'circle':
            self.xmin = -radius - ghost
            self.xmax = radius + ghost
            self.ymin = -radius - ghost
            self.ymax = radius +ghost
            self.zmin = zs_dipo - 50*unit
            self.zmax = ze_dipo + 50*unit
       
            lower_bound = [-radius, -radius, zs_dipo]
            upper_bound = [radius, radius, ze_dipo]
        elif chamber_type == 'crab':
            max_z = 500*unit
            l_main_y = 242*unit
            l_main_x = 300*unit
            l_main_z = 350*unit
            l_beam_pipe = 84*unit
            l_int = 62*unit
            l_main_int_y = l_main_y - 0.5*l_beam_pipe
            l_main_int_z = 0.5*l_main_z - l_int
            l_main_int_x = 0.5*l_main_x - l_int

            self.xmin = -0.5*l_main_x - ghost
            self.xmax = 0.5*l_main_x + ghost
            self.ymin = -0.5*l_main_y - ghost
            self.ymax = 0.5*l_main_y + ghost
            self.zmin = -0.5*max_z - 50*unit
            self.zmax = 0.5*max_z + 50 *unit

                   
     
            lower_bound = [-0.5*l_main_x, -0.5*l_main_y, -0.5*l_main_z]
            upper_bound = [0.5*l_main_x, 0.5*l_main_y, 0.5*l_main_z]

       ##########################
        # Beam parameters
        ##########################
        sigmaz = sigmat*picmi.clight
        if bunch_macro_particles > 0:
            bunch_w = bunch_intensity/bunch_macro_particles
        else:
            bunch_w = 0

        #######################################################
        # compute beam size from normalized emittance and beta
        # Uncomment if data available
        #######################################################
        bunch_rms_size            = [sigmax, sigmay, sigmaz]
        bunch_rms_velocity        = [0.,0.,0.]
        if chamber_type == 'crab':
            bunch_centroid_position   = [0,0,self.zmin+10*unit]
        else:
            bunch_centroid_position   = [0,0,zs_dipo-10*unit]
        
        bunch_centroid_velocity   = [0.,0.,beam_gamma*picmi.constants.c]

        self.beam = picmi.Species(particle_type = 'proton',
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
            vx0 = dict_init_dist['vx_mp'][0]
            vy0 = dict_init_dist['vy_mp'][0]
            vz0 = dict_init_dist['vz_mp'][0]
            w0 = dict_init_dist['nel_mp'][0]
            # compute the velocities
            #invgamma = np.sqrt(1-picmi.clight**2/(np.square(ux0)+np.square(uy0)+np.square(uz0)))
            #vx0 = np.multiply(invgamma,ux0)
            #vy0 = np.multiply(invgamma,uy0)
            #vz0 = np.multiply(invgamma,uy0)
            # Reload the outputs and other auxiliary stuff
            if flag_output:
                self.numelecs = dict_init_dist['numelecs'][0]
                self.N_mp = dict_init_dist['N_mp'][0]
                self.numelecs_tot = dict_init_dist['numelecs_tot'][0]
                self.xhist = dict_init_dist['xhist']
                self.bins = dict_init_dist['bins']
                self.b_pass_prev = dict_init_dist['b_pass'] -1
        else:
            if chamber_type == 'circle':
                x0 = random.uniform(lower_bound[0],upper_bound[0],init_num_elecs_mp)
                y0 = random.uniform(lower_bound[1],upper_bound[1],init_num_elecs_mp)
                z0 = random.uniform(lower_bound[2],upper_bound[2],init_num_elecs_mp)
                vx0 = np.zeros(init_num_elecs_mp)
                vy0 = np.zeros(init_num_elecs_mp)
                vz0 = np.zeros(init_num_elecs_mp)

                r0_sq = np.square(x0) + np.square(y0) 
                flag_out = np.where(r0_sq > radius*radius)[0]
                Nout = len(flag_out)
                while Nout>0:
                    x0[flag_out] = random.uniform(lower_bound[0],upper_bound[0],Nout)
                    y0[flag_out] = random.uniform(lower_bound[1],upper_bound[1],Nout)

                    r0_sq = np.square(x0) + np.square(y0)
                    flag_out = np.where(r0_sq>radius*radius)[0]
                    Nout = len(flag_out)

                w0 = float(init_num_elecs)/float(init_num_elecs_mp)            
            elif chamber_type == 'crab':
                x0 = random.uniform(lower_bound[0],upper_bound[0],init_num_elecs_mp)
                y0 = random.uniform(lower_bound[1],upper_bound[1],init_num_elecs_mp)
                z0 = random.uniform(lower_bound[2],upper_bound[2],init_num_elecs_mp)
                vx0 = np.zeros(init_num_elecs_mp)
                vy0 = np.zeros(init_num_elecs_mp)
                vz0 = np.zeros(init_num_elecs_mp)

                flag_out = np.where(np.logical_and(np.logical_and(np.logical_and(x0 < l_main_int_x, x0 > -l_main_int_x),
                                np.logical_and(z0 < l_main_int_z, z0 > -l_main_int_z)),
                             np.logical_or(y0 > 0.5*l_beam_pipe, y0 < -0.5*l_beam_pipe)))[0]
                Nout = len(flag_out)
                while Nout>0:
                    x0[flag_out] = random.uniform(lower_bound[0],upper_bound[0],Nout)
                    y0[flag_out] = random.uniform(lower_bound[1],upper_bound[1],Nout)
                    z0[flag_out] = random.uniform(lower_bound[2],upper_bound[2],Nout)

                    flag_out = np.where(np.logical_and(np.logical_and(np.logical_and(x0 < l_main_int_x, x0 > -l_main_int_x),
                                np.logical_and(z0 < l_main_int_z, z0 > -l_main_int_z)),
                             np.logical_or(y0 > 0.5*l_beam_pipe, y0 < -0.5*l_beam_pipe)))[0]
                     
                    Nout = len(flag_out)

                w0 = float(init_num_elecs)/float(init_num_elecs_mp)

            else:
                x0 = random.uniform(lower_bound[0],upper_bound[0],init_num_elecs_mp)
                y0 = random.uniform(lower_bound[1],upper_bound[1],init_num_elecs_mp)
                z0 = random.uniform(lower_bound[2],upper_bound[2],init_num_elecs_mp)
                vx0 = np.zeros(init_num_elecs_mp)
                vy0 = np.zeros(init_num_elecs_mp)
                vz0 = np.zeros(init_num_elecs_mp)
                w0 = float(init_num_elecs)/float(init_num_elecs_mp)


            self.b_pass_prev = 0

        electron_background_dist = picmi.ParticleListDistribution(x=x0, y=y0,
                                                                  z=z0, vx=vx0,
                                                                  vy=vy0, vz=vz0,
                                                                  weight=w0)

        self.elecb = picmi.Species(particle_type = 'electron',
                              particle_shape = 'linear',
                              name = 'Electron background',
                              initial_distribution = electron_background_dist)

        self.secelec = picmi.Species(particle_type = 'electron',
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


        grid = picmi.Cartesian3DGrid(number_of_cells = [self.nx, self.ny, self.nz],
                                     lower_bound = [self.xmin, self.ymin, self.zmin],
                                     upper_bound = [self.xmax, self.ymax, self.zmax],
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
            print('Using rectangular chamber with xaper: %1.2e, yaper: %1.2e' %(width/2., height/2.))
            upper_box = picmi.warp.YPlane(y0=height/2.,ysign=1,condid=1)
            lower_box = picmi.warp.YPlane(y0=-height/2.,ysign=-1,condid=1)
            left_box = picmi.warp.XPlane(x0=width/2.,xsign=1,condid=1)
            right_box = picmi.warp.XPlane(x0=-width/2.,xsign=-1,condid=1)

            sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                               warp_initialize_solver_after_generate = 1)

            sim.conductors = upper_box + lower_box + left_box + right_box

        elif chamber_type == 'LHC':
            print('Using the LHC chamber')
            #pipe_annulus = picmi.warp.ZAnnulus(rmin = radius, rmax    = radius+ghost/2, length  = l, condid =1)
            upper_box = picmi.warp.YPlane(y0=height/2,ysign=1,condid=1)
            lower_box = picmi.warp.YPlane(y0=-height/2,ysign=-1,condid=1)
            pipe = picmi.warp.ZCylinderOut(radius = radius, length = z_length,condid = 1)

            sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                                   warp_initialize_solver_after_generate = 1)

            sim.conductors = pipe+upper_box+lower_box

        elif chamber_type == 'circle':
            print('Using a circular chamber with radius %1.2e' %radius)
            pipe_annulus = picmi.warp.ZAnnulus(rmin = radius, rmax    = radius+ghost, length  = l,
                                voltage = 0., xcent   = 0., ycent   = 0.,
                                zcent   = 0., condid  = 1)
            pipe = picmi.warp.ZCylinderOut(radius = radius, length = l,condid = 1) 

            sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                                   warp_initialize_solver_after_generate = 1)

            sim.conductors = pipe_annulus
        
        elif chamber_type == 'crab':
            print('Simulating ECLOUD in a crab cavity')
            box1 = picmi.warp.Box(zsize=self.zmax-self.zmin, xsize=self.xmax-self.xmin,ysize=self.ymax-self.ymin)
            box2 = picmi.warp.Box(zsize=self.zmax-self.zmin, xsize=l_beam_pipe,ysize=l_beam_pipe)
            box3 = picmi.warp.Box(zsize=l_main_z, xsize=l_main_x,ysize=l_main_y)
            # Shape the cavity
            ycen1 = 0.5*l_beam_pipe+l_main_int_y
            ycen2 = -ycen1
            box4 = picmi.warp.Box(zsize=2*l_main_int_z, xsize=2*l_main_int_x, ysize=2*l_main_int_y, ycent=ycen1)
            box5 = picmi.warp.Box(zsize=2*l_main_int_z, xsize=2*l_main_int_x, ysize=2*l_main_int_y, ycent=ycen2)

            sim = picmi.Simulation(solver = solver, verbose = 1, cfl = 1.0,
                                   warp_initialize_solver_after_generate = 1)

            sim.conductors = box1-box2-box3+box4+box5

        sim.add_species(self.beam, layout=None,
                        initialize_self_field = solver=='EM')

        self.elecb_layout = picmi.PseudoRandomLayout(n_macroparticles=init_num_elecs_mp,
                                                seed = 3)

        sim.add_species(self.elecb, layout=self.elecb_layout,
                        initialize_self_field = solver=='EM')

        sim.add_species(self.secelec, layout=None, initialize_self_field=False)

        #########################
        # Add Dipole
        #########################
        if lattice_elem == 'dipole':
            picmi.warp.addnewdipo(zs = zs_dipo, ze = ze_dipo, by = By)
        elif lattice_elem == 'crab':
            [self.x,self.y,self.z,self.Ex,self.Ey,self.Ez] = picmi.getdatafromtextfile("efield.txt",nskip=1,dims=[6,None])
            [_,_,_,Hx,Hy,Hz] = picmi.getdatafromtextfile("hfield.txt",nskip=1,dims=[6,None])
            
            self.Bx = Hx*picmi.mu0
            self.By = Hy*picmi.mu0
            self.Bz = Hz*picmi.mu0

            ### Interpolate them at cell centers (as prescribed by Warp doc)
            self.d = abs(self.x[1]-self.x[0])
            ### Number of mesh cells
            self.NNx = int(round(2*np.max(self.x)/self.d))
            self.NNy = int(round(2*np.max(self.y)/self.d))
            self.NNz = int(round(2*np.max(self.z)/self.d))
            ### Number of mesh vertices
            self.nnx = self.NNx+1
            self.nny = self.NNy+1
            self.nnz = self.NNz+1

            self.Ex3d = self.Ex.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.Ey3d = self.Ey.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.Ez3d = self.Ez.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.Bx3d = self.Bx.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.By3d = self.By.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.Bz3d = self.Bz.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.x3d = self.x.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.y3d = self.y.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)
            self.z3d = self.z.reshape(self.nnz,self.nny,self.nnx).transpose(2,1,0)

            ### Rescale the fields at convenience
            maxE = 57*1e6
            kk = maxE/np.max(abs(self.Ey3d))
    
            self.Ex3d *= kk
            self.Ey3d *= kk
            self.Ez3d *= kk
            self.Bx3d *= kk
            self.By3d *= kk
            self.Bz3d *= kk

            self.Exx = 0.125*(self.Ex3d[0:-1,0:-1,0:-1]
                        + self.Ex3d[0:-1:,0:-1,1:]
                        + self.Ex3d[0:-1,1:,0:-1]
                        + self.Ex3d[0:-1,1:,1:]
                        + self.Ex3d[1:,1:,0:-1]
                        + self.Ex3d[1:,1:,1:]
                        + self.Ex3d[1:,0:-1,1:]
                        + self.Ex3d[1:,0:-1,0:-1])

            self.Eyy = 0.125*(self.Ey3d[0:-1,0:-1,0:-1]
                        + self.Ey3d[0:-1:,0:-1,1:]
                        + self.Ey3d[0:-1,1:,0:-1]
                        + self.Ey3d[0:-1,1:,1:]
                        + self.Ey3d[1:,1:,0:-1]
                        + self.Ey3d[1:,1:,1:]
                        + self.Ey3d[1:,0:-1,1:]
                        + self.Ey3d[1:,0:-1,0:-1])

            self.Ezz = 0.125*(self.Ez3d[0:-1,0:-1,0:-1]
                        + self.Ez3d[0:-1:,0:-1,1:]
                        + self.Ez3d[0:-1,1:,0:-1]
                        + self.Ez3d[0:-1,1:,1:]
                        + self.Ez3d[1:,1:,0:-1]
                        + self.Ez3d[1:,1:,1:]
                        + self.Ez3d[1:,0:-1,1:]
                        + self.Ez3d[1:,0:-1,0:-1])

            self.Bxx = 0.125*(self.Bx3d[0:-1,0:-1,0:-1]
                        + self.Bx3d[0:-1:,0:-1,1:]
                        + self.Bx3d[0:-1,1:,0:-1]
                        + self.Bx3d[0:-1,1:,1:]
                        + self.Bx3d[1:,1:,0:-1]
                        + self.Bx3d[1:,1:,1:]
                        + self.Bx3d[1:,0:-1,1:]
                        + self.Bx3d[1:,0:-1,0:-1])

            self.Byy = 0.125*(self.By3d[0:-1,0:-1,0:-1]
                        + self.By3d[0:-1:,0:-1,1:]
                        + self.By3d[0:-1,1:,0:-1]
                        + self.By3d[0:-1,1:,1:]
                        + self.By3d[1:,1:,0:-1]
                        + self.By3d[1:,1:,1:]
                        + self.By3d[1:,0:-1,1:]
                        + self.By3d[1:,0:-1,0:-1])

            self.Bzz = 0.125*(self.Bz3d[0:-1,0:-1,0:-1]
                        + self.Bz3d[0:-1:,0:-1,1:]
                        + self.Bz3d[0:-1,1:,0:-1]
                        + self.Bz3d[0:-1,1:,1:]
                        + self.Bz3d[1:,1:,0:-1]
                        + self.Bz3d[1:,1:,1:]
                        + self.Bz3d[1:,0:-1,1:]
                        + self.Bz3d[1:,0:-1,0:-1])

            self.xx = 0.125*(self.x3d[0:-1,0:-1,0:-1]
                        + self.x3d[0:-1:,0:-1,1:]
                        + self.x3d[0:-1,1:,0:-1]
                        + self.x3d[0:-1,1:,1:]
                        + self.x3d[1:,1:,0:-1]
                        + self.x3d[1:,1:,1:]
                        + self.x3d[1:,0:-1,1:]
                        + self.x3d[1:,0:-1,0:-1])
    
            self.yy = 0.125*(self.y3d[0:-1,0:-1,0:-1]
                      + self.y3d[0:-1:,0:-1,1:]
                      + self.y3d[0:-1,1:,0:-1]
                      + self.y3d[0:-1,1:,1:]
                      + self.y3d[1:,1:,0:-1]
                      + self.y3d[1:,1:,1:]
                      + self.y3d[1:,0:-1,1:]
                      + self.y3d[1:,0:-1,0:-1])
        
            self.zz = 0.125*(self.z3d[0:-1,0:-1,0:-1]
                      + self.z3d[0:-1:,0:-1,1:]
                      + self.z3d[0:-1,1:,0:-1]
                      + self.z3d[0:-1,1:,1:]
                      + self.z3d[1:,1:,0:-1]
                      + self.z3d[1:,1:,1:]
                      + self.z3d[1:,0:-1,1:]
                      + self.z3d[1:,0:-1,0:-1])


            

            ### Lattice spatial parameters
            self.zs = np.min(self.z3d)-self.d/2.
            self.ze = np.max(self.z3d)+self.d/2.
            self.xs = np.min(self.x3d)-self.d/2.
            self.ys = np.min(self.y3d)-self.d/2.
            ### Lattice temporal parameters
            self.Tf = 25e-9
            self.freq = 400*1e6
            self.Nt = 1000
            self.phase_disp=0
            self.time_array = np.linspace(0.,self.Tf,self.Nt)
            self.data_arrayE = np.sin(self.time_array*self.freq*2*np.pi+self.phase_disp)
            self.data_arrayB = np.sin(self.time_array*self.freq*2*np.pi-np.pi/2+self.phase_disp)

            ### Create overlapped lattice elements to have E and B in the same region
            self.ie,self.egrid = picmi.warp.addnewegrd(self.zs, self.ze,
                                                       dx=self.d, dy=self.d,
                                                       xs = self.xs,
                                                       ys = self.ys,
                                                       time=self.time_array,
                                                       data=self.data_arrayE,
                                                       ex=self.Ex3d,
                                                       ey=self.Ey3d,
                                                       ez=self.Ez3d)
                                             

            picmi.warp.addnewbgrd(self.zs, self.ze, dx=self.d, dy=self.d,
                                  xs = self.xs, ys = self.ys,
                                  time=self.time_array, data=self.data_arrayB,
                                  bx=self.Bx3d, by=self.By3d, bz=self.Bz3d)


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
            self.beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_gamma, w=bunch_w)

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

        sec.add(incident_species = self.elecb.wspecies,
                emitted_species  = self.secelec.wspecies,
                conductor        = sim.conductors)

        sec.add(incident_species = self.secelec.wspecies,
                emitted_species = self.secelec.wspecies,
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
            if l_force or self.n_step%self.stride_imgs==0:
                plt.close()
                (Nx,Ny,Nz) = np.shape(self.secelec.wspecies.get_density())
                fig, axs = plt.subplots(1, 2,figsize=(12, 4.5))
                fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
                d = self.secelec.wspecies.get_density() + self.elecb.wspecies.get_density() + self.beam.wspecies.get_density()
                d2 = self.secelec.wspecies.get_density()+ self.elecb.wspecies.get_density()
                im1 = axs[0].imshow(d[:,:,int(Nz/2)] .T,cmap='jet',origin='lower', vmin=0.2*np.min(d2[:,:,int(Nz/2)]), vmax=0.8*np.max(d2[:,:,int(Nz/2)]),extent=[self.xmin, self.xmax , self.ymin, self.ymax])
                axs[0].set_xlabel('x [m]')
                axs[0].set_ylabel('y [m]')
                axs[0].set_title('e- density')
                fig.colorbar(im1, ax=axs[0],)
                im2 = axs[1].imshow(d[int(Nx/2),:,:],cmap='jet',origin='lower', vmin=0.2*np.min(d2[int(Nx/2),:,:]), vmax=0.8*np.max(d2[int(Nx/2),:,:]),extent=[self.zmin, self.zmax , self.ymin, self.ymax], aspect = 'auto')
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

        self.ntsteps_p_bunch = b_spac/top.dt
        self.tot_nsteps = int(np.ceil(b_spac*(n_bunches)/top.dt))
        t_start = self.b_pass_prev*b_spac
        tstep_start = int(round(t_start/top.dt))


        # pre-allocate outputs
        if self.flag_output and not (self.flag_checkpointing and os.path.exists('temp_mps_info.mat')):
            self.numelecs = np.zeros(self.tot_nsteps)
            self.numelecs_tot = np.zeros(self.tot_nsteps)
            self.N_mp = np.zeros(self.tot_nsteps)
            self.xhist = np.zeros((n_bunches,nbins))

        dict_out = {}

        # aux variables
        self.b_pass = 0
        self.perc = 10
        self.t0 = time.time()
        
        # trapping warp std output
        self.text_trap = {True: StringIO(), False: sys.stdout}[enable_trap]
        self.original = sys.stdout
        self.n_step = 0
        
        #for n_step in range(tstep_start,tot_nsteps):
    def step(self, u_steps = 1):
        for u_step in range(u_steps):
            # if a passage is starting...
            if self.n_step/self.ntsteps_p_bunch >= self.b_pass+self.b_pass_prev:
               # Measure the duration of the previous passage
                self.b_pass+=1
                self.perc = 10
                if self.b_pass>1:
                    self.t_pass_1 = time.time()
                    self.t_pass = self.t_pass_1-self.t_pass_0
                self.t_pass_0 = time.time()
               # Perform regeneration if needed
                if self.secelec.wspecies.getn() > self.N_mp_max:
                    print('Number of macroparticles: %d' %(self.secelec.wspecies.getn()))
                    print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    
                    perform_regeneration(self.N_mp_target, self.secelec.wspecies, sec)
                 
                # Save stuff if checkpoint
                if self.flag_checkpointing and np.any(checkpoints == self.b_pass + self.b_pass_prev):
                    dict_out_temp = {}
                    print('Saving a checkpoint!')
                    secelec_w = self.secelec.wspecies.getw()
                    dict_out_temp['x_mp'] = np.concatenate((self.secelec.wspecies.getx(),self.elecb.wspecies.getx()))
                    dict_out_temp['y_mp'] = np.concatenate((self.secelec.wspecies.gety(),self.elecb.wspecies.gety()))
                    dict_out_temp['z_mp'] = np.concatenate((self.secelec.wspecies.getz(),self.elecb.wspecies.gety()))
                    dict_out_temp['vx_mp'] = np.concatenate((self.secelec.wspecies.getvx(),self.elecb.wspecies.getvx()))
                    dict_out_temp['vy_mp'] = np.concatenate((self.secelec.wspecies.getvy(),self.elecb.wspecies.getvy()))
                    dict_out_temp['vz_mp'] = np.concatenate((self.secelec.wspecies.getvz(),self.elecb.wspecies.getvz()))
                    dict_out_temp['nel_mp'] = np.concatenate((secelec_w,self.elecb.wspecies.getw()))
                    if flag_output:
                        dict_out_temp['numelecs'] = self.numelecs
                        dict_out_temp['numelecs_tot'] = self.numelecs_tot
                        dict_out_temp['N_mp'] = self.N_mp
                        dict_out_temp['xhist'] = self.xhist
                        dict_out_temp['bins'] = self.bins

                    dict_out_temp['b_pass'] = self.b_pass + self.b_pass_prev

                    filename = 'temp_mps_info.mat'

                    sio.savemat(filename, dict_out_temp)

                print('===========================')
                print('Bunch passage: %d' %(self.b_pass+self.b_pass_prev))
                print('Number of electrons in the dipole: %d' %(np.sum(self.secelec.wspecies.getw())+np.sum(self.elecb.wspecies.getw())))
                print('Number of macroparticles: %d' %(self.secelec.wspecies.getn()+self.elecb.wspecies.getn()))
                if self.b_pass > 1:
                    print('Previous passage took %ds' %self.t_pass)


            if self.n_step%self.ntsteps_p_bunch/self.ntsteps_p_bunch*100>self.perc:
                print('%d%% of bunch passage' %self.perc)
                self.perc = self.perc+10

            # Dump outputs
            if self.flag_output and self.n_step%self.stride_output==0:
                dict_out = {}
                dict_out['numelecs'] = self.numelecs
                dict_out['numelecs_tot'] = self.numelecs_tot
                dict_out['N_mp'] = self.N_mp
                # Compute the x-position histogram
                (self.xhist[self.b_pass-1], self.bins) = np.histogram(self.secelec.wspecies.getx(), range = (self.xmin,self.xmax), bins = self.nbins, weights = self.secelec.wspecies.getw(), density = False)
                dict_out['bins'] = self.bins
                dict_out['xhist'] = self.xhist
                sio.savemat(self.output_name, dict_out)

            # Perform a step
            sys.stdout = self.text_trap
            picmi.warp.step(1)
            sys.stdout = self.original
            #print(sum(self.secelec.wspecies.getw())+sum(self.elecb.wspecies.getw()))
            #if self.secelec.wspecies.getn()>0 and self.elecb.wspecies.getn()>0:
            #    print(max(max(np.sqrt(np.square(self.elecb.wspecies.getvx())+np.square(self.elecb.wspecies.getvy())+np.square(self.elecb.wspecies.getvz()))), max(np.sqrt(np.square(self.secelec.wspecies.getvx())+np.square(self.secelec.wspecies.getvy())+np.square(self.secelec.wspecies.getvz())))))
            # Store stuff to be saved
            if self.flag_output:
                secelec_w = self.secelec.wspecies.getw()
                elecb_w = self.elecb.wspecies.getw()
                elecs_density = self.secelec.wspecies.get_density(l_dividebyvolume=0)[:,:,int(self.nz/2.)] + self.elecb.wspecies.get_density(l_dividebyvolume=0)[:,:,int(self.nz/2.)]
                self.numelecs[self.n_step] = np.sum(elecs_density)
                elecs_density_tot = self.secelec.wspecies.get_density(l_dividebyvolume=0)[:,:,:] + self.elecb.wspecies.get_density(l_dividebyvolume=0)[:,:,:]
                self.numelecs_tot[self.n_step] = np.sum(elecs_density_tot)
                self.N_mp[self.n_step] = len(secelec_w)+len(elecb_w)
            
            self.n_step += 1

            if self.n_step > self.tot_nsteps:
                # Timer
                t1 = time.time()
                totalt = t1-self.t0
                # Dump outputs
                if self.flag_output:
                    dict_out['numelecs'] = self.numelecs
                    dict_out['N_mp'] = self.N_mp
                    dict_out['numelecs_tot'] = self.numelecs_tot
                    dict_out['xhist'] = self.xhist
                    dict_out['bins'] = self.bins
                    sio.savemat(self.output_name, dict_out)
        

                # Delete checkpoint if found
                if flag_checkpointing and os.path.exists('temp_mps_info.mat'):
                    os.remove('temp_mps_info.mat')

                print('Run terminated in %ds' %totalt)

    def all_steps(self):
        self.step(self.tot_nsteps)


