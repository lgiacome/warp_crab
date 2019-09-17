import numpy as np
#from pywarpx import picmi
from warp import picmi
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

max_z = 3000*unit
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
# physics components
##########################

# --- beam
beam_uz = 287
bunch_physical_particles  = 1.15e11
sigmax = 0.00017514074313652062*5
sigmay = 0.00018021833789232512*5
sigmaz = 7.55e-2
bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,-13.5*sigmaz]
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
                     #mass = 1.6726219e-27,
                     #charge = 1.60217662081e-19,
                     color='blue',
                     initial_distribution = gauss_dist)


##########################
# numerics components
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
                             upper_boundary_conditions = upper_boundary_conditions,
                             warpx_max_grid_size=32)

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
# diagnostics
##########################

# field_diag = picmi.FieldDiagnostic(grid = grid,
#                                    period = 20,
#                                    write_dir = 'diags')
# 
# part_diag = picmi.ParticleDiagnostic(period = 100,
#                                      species = [electrons, ions, beam],
#                                      write_dir = 'diags')
# 
# field_diag_lab = picmi.LabFrameFieldDiagnostic(grid = grid,
#                                                num_snapshots = 20,
#                                                dt_snapshots = 0.5*(zmax - zmin)/picmi.clight,
#                                                data_list = ["rho", "E", "B", "J"],
#                                                write_dir = 'lab_diags')
# 
# part_diag_lab = picmi.LabFrameParticleDiagnostic(grid = grid,
#                                                  num_snapshots = 20,
#                                                  dt_snapshots = 0.5*(zmax - zmin)/picmi.clight,
#                                                  species = [electrons, ions, beam],
#                                                  write_dir = 'lab_diags')

##########################
# simulation setup
##########################
box1 = picmi.warp.Box(zsize=zmax-zmin, xsize=xmax-xmin,ysize=ymax-ymin)
box2 = picmi.warp.Box(zsize=zmax-zmin, xsize=l_beam_pipe,ysize=l_beam_pipe)
box3 = picmi.warp.Box(zsize=l_main_z, xsize=l_main_x,ysize=l_main_y)
# Boxes for the poles
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
##########################
# Insert ext fields by adding lattice
##########################

############# READ GRID DATA FROM FILE
#### Get great data at mesh nodes

[x,y,z,Ex,Ey,Ez] = picmi.getdatafromtextfile("efield.txt",nskip=1,dims=[6,None])
[_,_,_,Hx,Hy,Hz] = picmi.getdatafromtextfile("hfield.txt",nskip=1,dims=[6,None])
Bx = Hx*picmi.mu0
By = Hy*picmi.mu0
Bz = Hz*picmi.mu0

### Interpolate them at cell centers
d = abs(x[1]-x[0])
#number of mesh cells
NNx = int(round(2*max(x)/d))
NNy = int(round(2*max(y)/d))
NNz = int(round(2*max(z)/d))
#number of mesh vertices
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
            ### Get the coords of the cell centers
            #xx[i,j,k] = x[i+nnx*j+(nnx*nny)*k]+x[(i+1)+nnx*j+(nnx*nny)*k]
            #yy[i,j,k] = y[i+nnx*j+(nnx*nny)*k]+y[i+nnx*(j+1)+(nnx*nny)*k]
            #zz[i,j,k] = z[i+nnx*j+(nnx*nny)*k]+z[i+nnx*j+(nnx*nny)*(k+1)]
            ### Get the fields at the cell centers
            for ii in [i,i+1]:
                for jj in [j,j+1]:
                    for kk in [k,k+1]:
                        Exx[i,j,k] += Ex[ii+nnx*jj+(nnx*nny)*kk]
                        Eyy[i,j,k] += Ey[ii+nnx*jj+(nnx*nny)*kk]
                        Ezz[i,j,k] += Ez[ii+nnx*jj+(nnx*nny)*kk]
                        Bxx[i,j,k] += Bx[ii+nnx*jj+(nnx*nny)*kk]
                        Byy[i,j,k] += By[ii+nnx*jj+(nnx*nny)*kk]
                        Bzz[i,j,k] += Bz[ii+nnx*jj+(nnx*nny)*kk]

#xx *= 0.5
#yy *= 0.5
#zz *= 0.523
Exx *= 0.125
Eyy *= 0.125
Ezz *= 0.125
Bxx *= 0.125
Byy *= 0.125
Bzz *= 0.125

zs = -659.9*unit
ze = 659.9*unit
xs = -l_main_x/2 + d/2
ys = -l_main_y/2 + d/2
############# INITIALIZE PARAMETERS
Tf = 25e-9
freq = 400*1e6
Nt = 100
def ss(t):
    print(np.sin((t-4.5*sigmaz/picmi.warp.clight)*freq*2*np.pi))

time_array = np.linspace(0.,Tf,Nt)
data_arrayE = np.sin((time_array-4.5*sigmaz/picmi.warp.clight)*freq*2*np.pi)
data_arrayB = np.sin((time_array-4.5*sigmaz/picmi.warp.clight)*freq*2*np.pi-np.pi/2)
############# CREATE THE OVERLAPPED LATTICE ELEMENTS
ie,egrid = picmi.warp.addnewegrd(zs, ze, dx=d, dy=d, xs = xs, ys = ys, nx=NNx, ny=NNy, nz=NNz, time=time_array, data=data_arrayE,
               ex=Exx, ey=Eyy, ez=Ezz)

picmi.warp.addnewbgrd(zs, ze, dx=d, dy=d, xs = xs, ys = ys, nx=NNx, ny=NNy, nz=NNz, time=time_array, data=data_arrayB,
           bx=Bxx, by=Byy, bz=Bzz)


##########################
# simulation run
##########################

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
#sim.write_input_file(file_name = 'inputs_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
#sim.step(max_steps)

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
#step(10)
