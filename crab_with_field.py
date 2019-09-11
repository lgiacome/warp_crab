import numpy as np
#from pywarpx import picmi
from warp import picmi

##########################
# physics parameters
##########################

mysolver = 'EM' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

# --- Beam
unit = 1e-3

beam_density = 1.e5
beam_uz = 0.1
beam_xmin = -30.*unit
beam_xmax = +30.*unit
beam_ymin = -30.*unit
beam_ymax = +30.*unit
beam_zmin = -30.*unit
beam_zmax = +30.*unit

##########################
# numerics parameters
##########################

# --- Nb time steps

max_steps = 1#500

# --- grid

nx = 100/2
ny = 100/2
nz = 100/2

max_z = 659.9*unit
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

# parabolic_beam = picmi.AnalyticDistribution(density_expression = "beam_density*np.maximum(0., (z - beam_zmin)*(beam_zmax - z)*4/(beam_zmax - beam_zmin)**2*(1. - (sqrt(x**2 + y**2)/beam_rmax)**2))",
parabolic_beam = picmi.AnalyticDistribution(density_expression = "beam_density*np.maximum(0., (z - beam_zmin)*(beam_zmax - z)*4/(beam_zmax - beam_zmin)**2*(1. - (sqrt(x**2)/beam_rmax)**2))",
                                            beam_density = beam_density,
                                            beam_rmax = beam_xmax,
                                            beam_zmin = beam_zmin,
                                            beam_zmax = beam_zmax,
                                            lower_bound = [beam_xmin, beam_ymin, beam_zmin],
                                            upper_bound = [beam_xmax, beam_ymax, beam_zmax],
                                            directed_velocity = [0., 0., beam_uz])

# parabolic_beam = None#picmi.ParticleListDistribution(x=0.,y=0.,z=0.,
# #                 ux=0.,uy=0.,uz=beam_uz*picmi.warp.clight,
# #                 weight=1.)

beam = picmi.Species(particle_type = 'proton',
                     particle_shape = 'linear',
                     name = 'beam',
                     mass = 1.6726219e-27,
                     charge = 1.60217662081e-19,
                     initial_distribution = parabolic_beam)

# 
# # define shortcuts
# pw = picmi.warp

# beam.wspecies.addpart(0.,0.,0.,0.,0.,100.*pw.clight,lmomentum=1)
# beam.wspecies.sw=1000.

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
#pipe = picmi.warp.ZAnnulus(rmin=xmax/4,rmax=xmax, length=1.,zcent=0.)

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

sim.add_species(beam, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=beam_number_per_cell_each_dim),
                                                 initialize_self_field = solver=='EM')

##########################
# Insert ext fields by adding lattice
##########################

############# READ GRID DATA FROM FILE
#### Get great data at mesh nodes

[x,y,z,Ex,Ey,Ez] = picmi.getdatafromtextfile("efield.txt",nskip=1,dims=[6,None])
[_,_,_,Hx,Hy,Hz] = picmi.getdatafromtextfile("hfield.txt",nskip=1,dims=[6,None])
Bx = picmi.mu0*Hx
By = picmi.mu0*Hy
Bz = picmi.mu0*Hz

### Interpolate them at cell centers
d = abs(x[1]-x[0])
#number of mesh cells
Nx = int(2*max(x)/d-1)
Ny = int(2*max(y)/d-1)
Nz = int(2*max(z)/d-1)
#number of mesh vertices
nx = int(2*max(x)/d)
ny = int(2*max(y)/d)
nz = int(2*max(z)/d)

xx = np.zeros([Nx,Ny,Nz],'d')
yy = np.zeros([Nx,Ny,Nz],'d')
zz = np.zeros([Nx,Ny,Nz],'d')
Exx = np.zeros([Nx,Ny,Nz],'d')
Eyy = np.zeros([Nx,Ny,Nz],'d')
Ezz = np.zeros([Nx,Ny,Nz],'d')
Bxx = np.zeros([Nx,Ny,Nz],'d')
Byy = np.zeros([Nx,Ny,Nz],'d')
Bzz = np.zeros([Nx,Ny,Nz],'d')

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            ### Get the coords of the cell centers
            xx[i,j,k] = x[i+nx*j+(nx+ny)*k]+x[(i+1)+nx*j+(nx+ny)*k]
            yy[i,j,k] = y[i+nx*j+(nx+ny)*k]+y[i+nx*(j+1)+(nx+ny)*k]
            zz[i,j,k] = z[i+nx*j+(nx+ny)*k]+z[i+nx*j+(nx+ny)*(k+1)]
            ### Get the fields at the cell centers
            for ii in [i,i+1]:
                for jj in [j,j+1]:
                    for kk in [k,k+1]:
                        Exx[i,j,k] += Ex[ii+nx*jj+(nx+ny)*kk]
                        Eyy[i,j,k] += Ey[ii+nx*jj+(nx+ny)*kk]
                        Ezz[i,j,k] += Ez[ii+nx*jj+(nx+ny)*kk]
                        Bxx[i,j,k] += Bx[ii+nx*jj+(nx+ny)*kk]
                        Byy[i,j,k] += By[ii+nx*jj+(nx+ny)*kk]
                        Bzz[i,j,k] += Bz[ii+nx*jj+(nx+ny)*kk]
xx *= 0.5
yy *= 0.5
zz *= 0.5
#Exx *= 0.125
#Eyy *= 0.125
#Ezz *= 0.125
#Bxx *= 0.125
#Byy *= 0.125
#Bzz *= 0.125

zs = -l_main_z/2
ze = l_main_z/2
############# INITIALIZE PARAMETERS
Tf = 25e-9
Nt = 20
time_array = np.linspace(0.,Tf,Nt)
data_arrayE = np.sin(time_array/Tf*2*np.pi)
data_arrayB = np.sin(time_array/Tf*2*np.pi-np.pi/2)
############# CREATE THE OVERLAPPED LATTICE ELEMENTS
picmi.addnewegrd(zs, ze, dx=d, dy=d, nx=Nx, ny=Ny, nz=Nz, time=time_array, data=data_arrayE,
               ex=Exx, ey=Eyy, ez=Ezz)
picmi.addnewbgrd(zs, ze, dx=d, dy=d, nx=Nx, ny=Ny, nz=Nz, time=time_array, data=data_arrayB,
               bx=Bxx, by=Byy, bz=Bzz)

# sim.add_diagnostic(field_diag)
# #sim.add_diagnostic(part_diag)
# sim.add_diagnostic(field_diag_lab)
# sim.add_diagnostic(part_diag_lab)

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

beam.wspecies.mass = 100.e10
beam.wspecies.sm = 100.e10
pg = beam.wspecies.pgroup

if mysolver=='ES':
    pw.top.dt = pw.w3d.dz/pw.ave(beam.wspecies.getvz())

def myplots(l_force=0):
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

        if mysolver=='EM':
            solver.solver.pfex(direction=1,l_transpose=1,view=9)
            solver.solver.pfez(direction=1,l_transpose=1,view=10)
        pw.refresh()

pw.installafterstep(myplots)

myplots(1)
#step(10)
