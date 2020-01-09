import numpy as np
import numpy.random as random
from warp import picmi
from warp.particles.Secondaries import *
import matplotlib.pyplot as plt
from mpi4py import MPI
##########################
# physics parameters
##########################

mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

# --- Beam
unit = 1e-3
dt = 25e-10
##########################
# numerics parameters
##########################

# --- grid
dh = .3e-3
ghost = 1e-3

zs_dipo = -500*unit
ze_dipo = 500*unit
r = 23.e-3
h = 18.e-3

xmin = -r - ghost
xmax = -xmin
ymin = -h - ghost
ymax = -ymin
zmin = zs_dipo- 50*unit
zmax = -zmin

nx = 254
ny = 221
nz = 24

l = ze_dipo-zs_dipo

#######################################################
# compute beam size from normalized emittance and beta
# Uncomment if data available
#######################################################

xin = np.array([-0.014313436500832114])
yin = np.array([-0.018799275505017324])
zin = np.array([-0.5017454920986667])

vx0 = np.array([-31174136.550982263])
vy0 = np.array([-39281343.76173671])
vz0 = np.array([-2140348.229993094])

x0 = xin - vx0*dt
y0 = yin - vy0*dt
z0 = zin - vz0*dt

magVsq = vx0**2+vy0**2+vz0**2

gammax = 1/np.sqrt(1-vx0**2/picmi.warp.clight**2)
gammay = 1/np.sqrt(1-vy0**2/picmi.warp.clight**2)
gammaz = 1/np.sqrt(1-vz0**2/picmi.warp.clight**2)

ux0 = vx0*gammax
uy0 = vy0*gammay
uz0 = vz0*gammaz

elec_gamma = 1/np.sqrt(1-magVsq/picmi.warp.clight**2)


electron_background_dist = picmi.ParticleListDistribution(x=x0, y=y0,
                                                          z=z0, ux=ux0,
                                                          uy=uy0, uz=uz0,
                                                          weight=1)

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

upper_box = picmi.warp.YPlane(y0=h,ysign=1,condid=1)
lower_box = picmi.warp.YPlane(y0=-h,ysign=-1,condid=1)

sim = picmi.Simulation(solver = solver,
                       verbose = 1,
                       cfl = 1.0,
                       warp_initialize_solver_after_generate = 1)

sim.conductors = pipe+upper_box+lower_box
#sim.conductors = pipe+lower_box+upper_box

sim.add_species(elecb, layout=None,
                initialize_self_field = solver=='EM')


sim.add_species(secelec, layout=None, initialize_self_field=False)


#########################
# Add Dipole
#########################

By = 0.53
picmi.warp.addnewdipo(zs = zs_dipo, ze = ze_dipo, by = By)


####################################
# Set material parameters from file
####################################

sim.step(1)
solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
sim.step(1)
    
pp = warp.ParticleScraper(sim.conductors,lsavecondid=1,lsaveintercept=1,lcollectlpdata=1)
        

sec=Secondaries(conductors=[sim.conductors])

sec.add(incident_species = elecb.wspecies,
                emitted_species  = elecb.wspecies,
                conductor        = sim.conductors,
                material = 'Cu')

# --- set weights of secondary electrons
#secelec.wspecies.getw()[...] = elecb.wspecies.getw()[0]

# define shortcuts
pw = picmi.warp
step=pw.step
if mysolver=='ES':
    #    pw.top.dt = pw.w3d.dz/pw.ave(beam.wspecies.getvz())
    pw.top.dt = dt #minnd([pw.w3d.dx,pw.w3d.dy,pw.w3d.dz])/clight

step(1)
