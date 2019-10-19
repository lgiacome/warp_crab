import numpy as np
import numpy.random as random
#from pywarpx import picmi
from warp import picmi
from scipy.stats import gaussian_kde
from warp.particles.Secondaries import *
import matplotlib.pyplot as plt
import parser
import scipy.io as sio
from io import BytesIO as StringIO
from mpi4py import MPI
##########################
# physics parameters
##########################

mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

# --- Beam

##########################
# numerics parameters
##########################

# --- grid

beam_number_per_cell_each_dim = [1, 1, 1]
dh = .5e-1
xmin = -1
xmax = 1
ymin = -1
ymax = 1
zmin = -1
zmax = 1

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
E = E0 + 300.1e3

beam_gamma = E/E0


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

wall = picmi.warp.YCylinderOut(0.5,1)

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
    print(np.shape(elec_beam.wspecies.getx()))

picmi.warp.installuserinjection(nonlinearsource)


##########################
# simulation run
##########################
sim.step(1)
solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
sim.step(1)


pp = warp.ParticleScraper(sim.conductors,lsavecondid=1,lsaveintercept=1,lcollectlpdata=1)

####################################
# Set material parameters from file
####################################

def set_params_user(maxsec, matnum):
    dict = parser.pos2dic('LHC_inj_72bx5.in')
    
    posC.matsurf = dict['matsurf']
    posC.iprob = dict['iprob']
    
    posC.enpar = dict['enpar']
    
    posC.pnpar= dict['pnpar']
    
    posC.dtspk = dict['dtspk']
    posC.dtotpk = dict['dtotpk']
    posC.pangsec = dict['pangsec']
    posC.pr = dict['pr']
    posC.sige = dict['sige']
    posC.Ecr = dict['Ecr']
    posC.E0tspk = dict['E0tspk']
    posC.E0epk = dict['E0epk']
    posC.E0w = dict['E0w']
    posC.rpar1 = dict['rpar'][0]
    posC.rpar2 = dict['rpar'][1]
    posC.tpar1 = dict['tpar'][0]
    posC.tpar2 = dict['tpar'][1]
    posC.tpar3 = dict['tpar'][2]
    posC.tpar4 = dict['tpar'][3]
    posC.tpar5 = dict['tpar'][4]
    posC.tpar6 = dict['tpar'][5]
    posC.epar1 = dict['epar'][0]
    posC.epar2 = dict['epar'][1]
    posC.P1rinf = dict['P1rinf']
    posC.P1einf = dict['P1einf']
    posC.P1epk = dict['P1epk']
    posC.powts = dict['powts']
    posC.powe = dict['powe']
    posC.qr = dict['qr']


sec=Secondaries(conductors=sim.conductors, set_params_user  = set_params_user,
                l_usenew=0)

sec.add(incident_species = elec_beam.wspecies,
        emitted_species  = secelec.wspecies,
        conductor        = sim.conductors)

# --- set weights of secondary electrons
#secelec.wspecies.getw()[...] = elecb.wspecies.getw()[0]

# define shortcuts
pw = picmi.warp
pw.winon()
em = solver.solver
step=pw.step

if mysolver=='ES':
    print(pw.ave(elec_beam.wspecies.getvz())/picmi.clight)

    #    pw.top.dt = pw.w3d.dz/pw.ave(beam.wspecies.getvz())
    pw.top.dt = 25e-11 #minnd([pw.w3d.dx,pw.w3d.dy,pw.w3d.dz])/clight


def myplots(l_force=0):
    if mysolver=='EM':  
        # why is this needed?
        em.clearconductors()
        em.installconductor(sim.conductors, dfill = picmi.warp.largepos)
    if l_force or pw.top.it%10==0:
        pw.fma()
        '''
        if 1==1: #mysolver=='ES':
            #solver.solver.pcphizx(filled=1,view=3,titles=0)
            #beam.wspecies.ppzx(filled=1,view=3,msize=2,color='red')
            #pw.limits(zmin,zmax,xmin,xmax)
            #solver.solver.pcphizy(filled=1,view=4,titles=0)
#            beam.wspecies.ppzy(filled=1,msize=1,color='red',width=3)
            pw.plsys(9)
            sim.conductors.drawxy()
            beam.wspecies.ppxy(msize=1,color='black')
            elecb.wspecies.ppxy(msize=1,color='cyan')
            secelec.wspecies.ppxy(msize=1,color='magenta')
            pw.limits(picmi.w3d.xmmin,picmi.w3d.xmmax,picmi.w3d.ymmin,picmi.w3d.ymmax)
            pw.plsys(10)
            #sim.conductors.drawzy()
            beam.wspecies.ppzy(msize=1,color='black')
            elecb.wspecies.ppzy(msize=1,color='cyan')
            secelec.wspecies.ppzy(msize=1,color='magenta')
            pw.limits(picmi.w3d.zmmin,picmi.w3d.zmmax,picmi.w3d.ymmin,picmi.w3d.ymmax)
#            pw.limits(np.min(pw.getz()),np.max(pw.getz()),np.min(pw.gety()),np.max(pw.gety()))
            #solver.solver.pcphixy(filled=1,view=5,titles=0)
            #beam.wspecies.ppxy(filled=1,view=5,msize=2,color='red')
        #pw.limits(xmin,xmax,ymin,ymax)
        #pw.plotegrd(ie,component = 'y', iz = 0, view = 6)
        '''
    if l_force or 0==0:#pw.top.it%1==0:
        plt.close()
        (Nx,Ny,Nz) = np.shape(secelec.wspecies.get_density())
        fig, axs = plt.subplots(1, 2,figsize=(12, 4.5))
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
        d = secelec.wspecies.get_density() + elec_beam.wspecies.get_density()
        im1 = axs[0].imshow(d[:,:,Nz/2] .T,cmap='jet',origin='lower', vmin=0.2*np.min(d[:,:,Nz/2]), vmax=0.8*np.max(d[:,:,Nz/2]),extent=[xmin, xmax , ymin, ymax])
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('y [m]')
        axs[0].set_title('e- density')
        fig.colorbar(im1, ax=axs[0],)
        im2 = axs[1].imshow(d[Nx/2,:,:],cmap='jet',origin='lower', vmin=0.2*np.min(d[Nx/2,:,:]), vmax=0.8*np.max(d[Nx/2,:,:]),extent=[zmin, zmax , ymin, ymax], aspect = 'auto')
        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title('e- density')
        fig.colorbar(im2, ax=axs[1])
        n_step = top.time/top.dt
        figname = 'images2/%d.png' %n_step
        plt.draw()
        plt.pause(1e-8)
        
    if 1==0: #mysolver=='EM':
        solver.solver.pfex(direction=1,l_transpose=1,view=9)
        solver.solver.pfez(direction=1,l_transpose=1,view=10)
    pw.refresh()


pw.installafterstep(myplots)

myplots(1)

n_step = 0
tot_nsteps = 0
numelecs = np.zeros(tot_nsteps)
total = np.zeros(tot_nsteps)
b_pass = 0
perc = 10
dict_out = {}
original = sys.stdout
text_trap = StringIO()
#sys.stdout = text_trap
t0 = time.time()
for n_step in range(tot_nsteps):
    step(1)


t1 = time.time()
totalt = t1-t0
dict_out['numelecs'] = numelecs
dict_out['total'] = total
sio.savemat('output0.mat',dict_out)

print('Run terminated in %ds' %totalt)

