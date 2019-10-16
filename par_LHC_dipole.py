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
unit = 1e-3

##########################
# numerics parameters
##########################

# --- grid
dh = .3e-3

zs_dipo = -500*unit
ze_dipo = 500*unit
r = 23.e-3
h = 18.e-3

xmin = -r
xmax = r
ymin = -r
ymax = r
zmin = zs_dipo-50*unit
zmax = ze_dipo+50*unit

nx = (xmax-xmin)/dh
ny = (xmax-xmin)/dh
nz = (xmax-xmin)/dh


l = zmax-zmin
sigmat= 1.000000e-09/4.
sigmaz = sigmat*299792458.
b_spac = 25e-9
t_offs = b_spac-6*sigmat
n_bunches = 1

beam_number_per_cell_each_dim = [1, 1, 1]

##########################
# Beam parameters
##########################

# --- beam
beam_uz = 479.
bunch_physical_particles  = 2.5e11
bunch_w = 1e6
bunch_macro_particles = bunch_physical_particles/bunch_w
#######################################################
# compute beam size from normalized emittance and beta
# Uncomment if data available
#######################################################
'''
enorm_x = 3.5e-7
enorm_y = 3.5e-7
beta_x = 40
beta_y = 80
gamma_rel = beam_uz
beta_rel = 1/np.sqrt(1+1./(gamma_rel**2))
egeom_x = enorm_x/(beta_rel*gamma_rel)
egeom_y = enorm_y/(beta_rel*gamma_rel)
sigmax = np.sqrt(egeom_x*beta_x)
sigmay = np.sqrt(egeom_y*beta_y)
'''

sigmax = 2e-4
sigmay = 2.1e-4

bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,zs_dipo-10*unit]
bunch_centroid_velocity   = [0.,0.,beam_uz*picmi.constants.c]
gauss_dist = picmi.GaussianBunchDistribution(
                                            n_physical_particles = bunch_physical_particles,
                                            rms_bunch_size       = bunch_rms_size,
                                            rms_velocity         = bunch_rms_velocity,
                                            centroid_position    = bunch_centroid_position,
                                            centroid_velocity    = bunch_centroid_velocity )

beam = picmi.Species(particle_type = 'proton',
                     particle_shape = 'linear',
                     name = 'beam')
                     #initial_distribution = gauss_dist)

electron_background_dist = picmi.UniformDistribution(
                                                     lower_bound = [-r,
                                                                    -h,
                                                                     zs_dipo],
                                                     upper_bound = [ r,
                                                                     h,
                                                                     ze_dipo],
                                                     density = 1.e5/0.0014664200235342726
                                                     )

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

pipe = picmi.warp.ZCylinderOut(r,l,condid=1)
ycent_upper =  h + 0.5*(ymax+h)
ycent_lower =  h - 0.5*(ymin+h)

#upper_box = picmi.warp.Box(zsize=zmax-zmin, xsize=xmax-xmin,ysize=ymax-h, ycent = h + ycent_upper)
#lower_box = picmi.warp.Box(zsize=zmax-zmin, xsize=xmax-xmin,ysize=ymax-h, ycent = - h - ycent_lower)

upper_box = picmi.warp.YPlane(y0=h,ysign=1,condid=1)
lower_box = picmi.warp.YPlane(y0=-h,ysign=-1,condid=1)

sim = picmi.Simulation(solver = solver,
                       max_steps = max_steps,
                       verbose = 1,
                       cfl = 1.0,
                       warp_initialize_solver_after_generate = 1)

sim.conductors = pipe+upper_box+lower_box

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

sim.add_species(beam, layout=beam_layout,
                initialize_self_field = solver=='EM')

elecb_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

sim.add_species(elecb, layout=elecb_layout,
                initialize_self_field = solver=='EM')

sim.add_species(secelec, layout=None, initialize_self_field=False)
                
#########################
# Add Dipole
#########################

By = 0.53
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
    z = bunch_centroid_position[2]#+random.uniform(-0.5,0.5,NP)*top.dt*picmi.warp.clight*np.sqrt(1-1./(beam_uz**2))
    vx = random.normal(bunch_centroid_velocity[0],bunch_rms_velocity[0],NP)
    vy = random.normal(bunch_centroid_velocity[1],bunch_rms_velocity[1],NP)
    vz = picmi.warp.clight*np.sqrt(1-1./(beam_uz**2))
    beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_uz, w=bunch_w)

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
    dict = parser.pos2dic('2.20.txt')

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

# --- set emission of neutrals
sec=Secondaries(conductors=sim.conductors, set_params_user  = set_params_user)
sec.add(incident_species = elecb.wspecies,
        emitted_species  = secelec.wspecies,
        conductor        = sim.conductors)
sec.add(incident_species = secelec.wspecies,
         emitted_species = secelec.wspecies,
         conductor       = sim.conductors)

# --- set weights of secondary electrons
#secelec.wspecies.getw()[...] = elecb.wspecies.getw()[0]

# define shortcuts
pw = picmi.warp
pw.winon()
em = solver.solver
step=pw.step

if mysolver=='ES':
    print(pw.ave(beam.wspecies.getvz())/picmi.clight)
#    pw.top.dt = pw.w3d.dz/pw.ave(beam.wspecies.getvz())
    pw.top.dt = minnd([pw.w3d.dx,pw.w3d.dy,pw.w3d.dz])/clight

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
    #   plt.draw()
    #plt.pause(1e-8)
        
    if 1==0: #mysolver=='EM':
        solver.solver.pfex(direction=1,l_transpose=1,view=9)
        solver.solver.pfez(direction=1,l_transpose=1,view=10)
    pw.refresh()
        


def myplots2(l_force=0):
    if mysolver=='EM':
        # why is this needed?
        em.clearconductors()
        em.installconductor(sim.conductors, dfill = picmi.warp.largepos)
    if l_force or pw.top.it%1==0:
        pw.fma()
        if mysolver=='ES':
            solver.solver.pcselfexy('x')
            pw.limits(zmin,zmax,xmin,xmax)
        
        if mysolver=='EM':
            solver.solver.pfex(direction=1,l_transpose=1,view=9)
            solver.solver.pfez(direction=1,l_transpose=1,view=10)
        pw.refresh()

#pw.installafterstep(myplots)

#myplots(1)

ntsteps_p_bunch = b_spac/top.dt
n_step = 0
tot_nsteps = int(round(b_spac*n_bunches/top.dt))
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
    ts0 = time.time()
    if n_step/ntsteps_p_bunch > b_pass:
        b_pass+=1
        perc = 10
        print('===========================')
        print('Bunch passage: %d' %b_pass)
        print('Number of electrons in the dipole: %d' %(np.sum(secelec.wspecies.getw())+np.sum(elecb.wspecies.getw())))
    if n_step%ntsteps_p_bunch/ntsteps_p_bunch*100>=perc:
        print('%d%% of bunch passage' %perc)
        perc = perc+10
    original = sys.stdout
    sys.stdout = text_trap
    step(1)
    numelecs[n_step] = np.sum(secelec.wspecies.getw())+np.sum(elecb.wspecies.getw())
    ts1 = time.time()
    total[n_step] = ts1-ts0
    sys.stdout = original
    #print(numelecs[n_step])
#   if n_step%10==0:
#        dict_out['numelecs'] = numelecs
#        sio.savemat('output.mat',dict_out)

dict_out['numelecs'] = numelecs
dict_out['total'] = total
sio.savemat('output.mat',dict_out)
t1 = time.time()
totalt = t1 - t0
print('Run terminated in %ds' %totalt)


########################
# My plots
########################
'''
fig, axes = plt.subplots(nrows=1, ncols=2)
plt.subplot(1,2,1)
zz1 = pw.getz().copy()
yy1 = pw.gety().copy()
xy1 = np.vstack([zz1/pw.clight,yy1*1e3])
z1 = gaussian_kde(xy1)(xy1)
plt.scatter((zz1/pw.clight-np.mean(zz1/pw.clight))*1e9, yy1*1e3 -  np.mean(yy1*1e3),
        c=z1/np.max(z1), cmap='jet',  s=100, edgecolor='')
plt.xlim(-1.2,1.2)
plt.ylim(-6,6)
plt.xlabel('t [ns]')
plt.ylabel('y [mm]')

if mysolver == 'EM':
    step(650)
if mysolver == 'ES':
    step(100)

#plt.subplot(1,2,2)

zz2 = pw.getz().copy()
yy2 = pw.gety().copy()
xy2 = np.vstack([zz2/pw.clight,yy2*1e3])
z2 = gaussian_kde(xy2)(xy2)
sc = plt.scatter((zz2/pw.clight-np.mean(zz2/pw.clight))*1e9, yy2*1e3 -  np.mean(yy2*1e3),
            c=z2/np.max(z2), cmap='jet',  s=100, edgecolor='')
plt.xlim(-1.2,1.2)
plt.ylim(-6,6)
plt.xlabel('t [ns]')
plt.ylabel('y [mm]')

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(sc, cax=cbar_ax)
plt.colorbar()
plt.show()
'''
