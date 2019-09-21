import numpy as np
#from pywarpx import picmi
from warp import picmi
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
nz = 400/2

max_z = 7000*unit
l_main_y = 242*unit
l_main_x = 300*unit
l_main_z = 350*unit
l_beam_pipe = 84*unit
l_int = 62*unit
l_main_int_y = l_main_y - 0.5*l_beam_pipe
l_main_int_z = 0.5*l_main_z - l_int
l_main_int_x = 0.5*l_main_x - l_int

xmin = -0.5*l_main_x
xmax = 0.5*l_main_x
ymin = -0.5*l_main_y
ymax = 0.5*l_main_y
zmin = -0.5*max_z
zmax = 0.5*max_z

beam_number_per_cell_each_dim = [1, 1, 1]

##########################
# Beam parameters
##########################

# --- beam
beam_uz = 28.7
bunch_physical_particles  = 0.8e11

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

sigmax = 1.85e-3
sigmay = 1.85e-3

sigmaz = 1.85e-1
bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,-8*sigmaz]
bunch_centroid_velocity   = [0.,0.,beam_uz*picmi.constants.c]
gauss_dist = picmi.GaussianBunchDistribution(
                                            n_physical_particles = bunch_physical_particles,
                                            rms_bunch_size       = bunch_rms_size,
                                            rms_velocity         = bunch_rms_velocity,
                                            centroid_position    = bunch_centroid_position,
                                            centroid_velocity    = bunch_centroid_velocity )

beam = picmi.Species(particle_type = 'proton',
                     particle_shape = 'linear',
                     name = 'beam',
                     initial_distribution = gauss_dist)


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
box1 = picmi.warp.Box(zsize=zmax-zmin, xsize=xmax-xmin,ysize=ymax-ymin)
box2 = picmi.warp.Box(zsize=zmax-zmin, xsize=l_beam_pipe,ysize=l_beam_pipe)
box3 = picmi.warp.Box(zsize=l_main_z, xsize=l_main_x,ysize=l_main_y)
# Shape the cavity
ycen1 = 0.5*(l_beam_pipe+l_main_int_y)
ycen2 = -ycen1
box4 = picmi.warp.Box(zsize=l_main_int_z, xsize=l_main_int_x, ysize=l_main_int_y, ycent=ycen1)
box5 = picmi.warp.Box(zsize=l_main_int_z, xsize=l_main_int_x, ysize=l_main_int_y, ycent=ycen2)

sim = picmi.Simulation(solver = solver,
                       max_steps = max_steps,
                       verbose = 1,
                       cfl = 1.0,
                       warp_initialize_solver_after_generate = 1)

sim.conductors = box1-box2-box3+box4+box5

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

sim.add_species(beam, layout=beam_layout,
                initialize_self_field = solver=='EM')

#####################################################
# Insert RF fields by adding lattice components
#####################################################

### Get data at mesh nodes
[x,y,z,Ex,Ey,Ez] = picmi.getdatafromtextfile("efield.txt",nskip=1,dims=[6,None])
[_,_,_,Hx,Hy,Hz] = picmi.getdatafromtextfile("hfield.txt",nskip=1,dims=[6,None])
Bx = Hx*picmi.mu0
By = Hy*picmi.mu0
Bz = Hz*picmi.mu0

### Interpolate them at cell centers (how prescribed by Warp doc)
d = abs(x[1]-x[0])
### Number of mesh cells
NNx = int(round(2*max(x)/d))
NNy = int(round(2*max(y)/d))
NNz = int(round(2*max(z)/d))
### Number of mesh vertices
nnx = NNx+1
nny = NNy+1
nnz = NNz+1

xx = np.zeros([NNx,NNy,NNz],'d')
yy = np.zeros([NNx,NNy,NNz],'d')
zz = np.zeros([NNx,NNy,NNz],'d')
Exx = np.zeros([NNx,NNy,NNz],'d')
Eyy = np.zeros([NNx,NNy,NNz],'d')
Ezz = np.zeros([NNx,NNy,NNz],'d')
Bxx = np.zeros([NNx,NNy,NNz],'d')
Byy = np.zeros([NNx,NNy,NNz],'d')
Bzz = np.zeros([NNx,NNy,NNz],'d')

for i in range(NNx):
    for j in range(NNy):
        for k in range(NNz):
            for ii in [i,i+1]:
                for jj in [j,j+1]:
                    for kk in [k,k+1]:
                        Exx[i,j,k] += Ex[ii+nnx*jj+(nnx*nny)*kk]
                        Eyy[i,j,k] += Ey[ii+nnx*jj+(nnx*nny)*kk]
                        Ezz[i,j,k] += Ez[ii+nnx*jj+(nnx*nny)*kk]
                        Bxx[i,j,k] += Bx[ii+nnx*jj+(nnx*nny)*kk]
                        Byy[i,j,k] += By[ii+nnx*jj+(nnx*nny)*kk]
                        Bzz[i,j,k] += Bz[ii+nnx*jj+(nnx*nny)*kk]

Exx *= 0.125
Eyy *= 0.125
Ezz *= 0.125
Bxx *= 0.125
Byy *= 0.125
Bzz *= 0.125

### Rescale the fields at convenience
max = 57*1e6
kk = -max/Eyy[32,25,141]

Exx *= kk
Eyy *= kk
Ezz *= kk
Bxx *= kk
Byy *= kk
Bzz *= kk

### Lattice spatial parameters
zs = -659.9*unit
ze = 659.9*unit
xs = -l_main_x/2 + d/2
ys = -l_main_y/2 + d/2
### Lattice temporal parameters
Tf = 25e-9
freq = 400*1e6
Nt = 1000

time_array = np.linspace(0.,Tf,Nt)
data_arrayE = np.sin((time_array-4*sigmaz/picmi.warp.clight)*freq*2*np.pi)
data_arrayB = np.sin((time_array-4*sigmaz/picmi.warp.clight)*freq*2*np.pi-np.pi/2)

### Create overlapped lattice elements to have E and B in the same region
ie,egrid = picmi.warp.addnewegrd(zs, ze, dx=d, dy=d, xs = xs, ys = ys, nx=NNx, ny=NNy, nz=NNz, time=time_array,
                                 data=data_arrayE, ex=Exx, ey=Eyy, ez=Ezz)

picmi.warp.addnewbgrd(zs, ze, dx=d, dy=d, xs = xs, ys = ys, nx=NNx, ny=NNy, nz=NNz, time=time_array, data=data_arrayB,
           bx=Bxx, by=Byy, bz=Bzz)


##########################
# simulation run
##########################
sim.step(1)
solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
sim.step(1)

# define shortcuts
pw = picmi.warp
pw.winon()
em = solver.solver
step=pw.step

if mysolver=='ES':
    print(pw.ave(beam.wspecies.getvz())/picmi.clight)
    pw.top.dt = pw.w3d.dz/pw.ave(beam.wspecies.getvz())

def myplots(ie=0, l_force=0):
    if mysolver=='EM':  
        # why is this needed?
        em.clearconductors()
        em.installconductor(sim.conductors, dfill = picmi.warp.largepos)
    if l_force or pw.top.it%1==0:
        pw.fma()
        if mysolver=='ES':
            solver.solver.pcphizx(filled=1,view=3,titles=0)
            beam.wspecies.ppzx(msize=2,color='red')
            pw.limits(zmin,zmax,xmin,xmax)
            solver.solver.pcphizy(filled=1,view=4,titles=0)
            beam.wspecies.ppzy(msize=2,color='red')
            pw.limits(zmin,zmax,ymin,ymax)
            solver.solver.pcphixy(filled=1,view=5,titles=0)
            beam.wspecies.ppxy(msize=2,color='red')
            pw.limits(xmin,xmax,ymin,ymax)
            pw.plotegrd(ie,component = 'y', iz = 0, view = 6)
        
        if mysolver=='EM':
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

myplots(ie,1)


########################
# My plots
########################
#fig, axes = plt.subplots(nrows=1, ncols=2)
#plt.subplot(1,2,1)
#zz1 = pw.getz().copy()
#yy1 = pw.gety().copy()
#xy1 = np.vstack([zz1/pw.clight,yy1*1e3])
#z1 = gaussian_kde(xy1)(xy1)
#plt.scatter((zz1/pw.clight-np.mean(zz1/pw.clight))*1e9, yy1*1e3 -  np.mean(yy1*1e3),
#        c=z1/np.max(z1), cmap='jet',  s=100, edgecolor='')
#plt.xlim(-1.2,1.2)
#plt.ylim(-6,6)
#plt.xlabel('t [ns]')
#plt.ylabel('y [mm]')

step(90)

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

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(sc, cax=cbar_ax)
plt.colorbar()
plt.show()

