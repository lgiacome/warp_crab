import numpy as np
#from pywarpx import picmi
from warp import picmi

##########################
# physics parameters
##########################

mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)

# --- Beam
unit = 1e-3

beam_density = 1.e16
beam_uz = 100.
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
l_main_x = 248*unit
l_main_y = 350*unit
l_main_z = 300*unit
l_beam_pipe = 84*unit
l_int = 62*unit
l_main_int_x = l_main_x - 0.5*l_beam_pipe
l_main_int_z = 0.5*l_main_z - l_int
l_main_int_y = 0.5*l_main_y - l_int

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
                     mass = 1.e10,
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
xcen1 = 0.5*(l_beam_pipe+l_main_int_x)
xcen2 = -xcen1
box4 = picmi.warp.Box(zsize=l_main_int_z, xsize=l_main_int_x, ysize=l_main_int_y, xcent=xcen1)
box5 = picmi.warp.Box(zsize=l_main_int_z, xsize=l_main_int_x, ysize=l_main_int_y, xcent=xcen2)

sim = picmi.Simulation(solver = solver,
                       max_steps = max_steps,
                       verbose = 1,
                       cfl = 1.0,
                       warp_initialize_solver_after_generate = 1)

sim.conductors = box1-box2-box3+box4+box5

sim.add_species(beam, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=beam_number_per_cell_each_dim),
                                                 initialize_self_field = solver=='EM')

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
