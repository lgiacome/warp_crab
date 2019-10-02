import numpy as np
import numpy.random as random
#from pywarpx import picmi
from warp import picmi
from scipy.stats import gaussian_kde
from warp.particles.Secondaries import *
import matplotlib.pyplot as plt

##########################
# physics parameters
##########################

mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

# --- Beam
unit = 1e-3

##########################
# numerics parameters
##########################

# --- Nb time steps

max_steps = 1#500

# --- grid

nx = 100/2
ny = 100/2
nz = 100/2

max_z = 7000*unit

xmin = -23e-3
xmax = 23e-3
ymin = -23e-3
ymax = 23e-3
zmin = 0
zmax = max_z
r = 23.e-3
h = 18.e-3
l = zmax-zmin
sigmaz = 1.000000e-09/4.*299792458.

zs_dipo = 2000*unit
ze_dipo = 2100*unit

beam_number_per_cell_each_dim = [1, 1, 1]

##########################
# Beam parameters
##########################

# --- beam
beam_uz = 479.
bunch_physical_particles  = 1e11

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
sigmay = 2e-4

bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,1.5]
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
                                                     density = 1.e5
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

pipe = picmi.warp.ZCylinderOut(r,l)
ycent_upper =  h + 0.5*(ymax+h)
ycent_lower =  h - 0.5*(ymin+h)

upper_box = picmi.warp.Box(zsize=zmax-zmin, xsize=xmax-xmin,ysize=ymax-h, ycent = h + ycent_upper)
lower_box = picmi.warp.Box(zsize=zmax-zmin, xsize=xmax-xmin,ysize=ymax-h, ycent = - h - ycent_lower)

upper_box = picmi.warp.YPlane(y0=h,ysign=1)
lower_box = picmi.warp.YPlane(y0=-h,ysign=-1)

sim = picmi.Simulation(solver = solver,
                       max_steps = max_steps,
                       verbose = 1,
                       cfl = 1.0,
                       warp_initialize_solver_after_generate = 1)

sim.conductors = upper_box+pipe+lower_box

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
    b_spac = 25e-9
    n_bunches = 3
    val = 0
    sigmat = sigmaz/picmi.clight
    for i in range(1,n_bunches+1):
        val += bunch_physical_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac+0.8*b_spac)*(t-i*b_spac+0.8*b_spac)/(2*sigmat*sigmat))*picmi.warp.top.dt
    return val

def nonlinearsource():
    print("Generating paritcles")
    NP = int(time_prof(top.time))
    print(time_prof(top.time))
    x = random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
    y = random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
    z = bunch_centroid_position[2]
    vx = random.normal(bunch_centroid_velocity[0],bunch_rms_velocity[0],NP)
    vy = random.normal(bunch_centroid_velocity[1],bunch_rms_velocity[1],NP)
    vz = picmi.warp.clight*np.sqrt(1-1./(beam_uz**2))
    print("Adding paritcles")
    beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_uz)
    print("Done with generation")


picmi.warp.installuserinjection(nonlinearsource)


##########################
# simulation run
##########################
sim.step(1)
solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
sim.step(1)

pp = warp.ParticleScraper(sim.conductors,lsavecondid=1,lsaveintercept=1,lcollectlpdata=1)

# --- set emission of neutrals
sec=Secondaries(conductors=[sim.conductors])
sec.add(incident_species = elecb.wspecies,
        emitted_species  = secelec.wspecies,
        conductor        = sim.conductors,
        material         = 'Cu')
sec.add(incident_species = secelec.wspecies,
         emitted_species = secelec.wspecies,
         conductor       = sim.conductors,
         material        = 'Cu')

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
    if l_force or pw.top.it%100==0:
        pw.fma()
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

pw.installafterstep(myplots)

myplots(1)


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
