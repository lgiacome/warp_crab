from warp import picmi
import numpy as np

class Dipole:
    
    def __init__(self, zs_dipo, ze_dipo, By):
        picmi.warp.addnewdipo(zs = zs_dipo, ze = ze_dipo, by = By)

class CrabFields:

    def __init(self, max_rescale = 1):
        get_data = picmi.getdatafromtextfile
        [self.x,self.y,self.z,self.Ex,self.Ey,self.Ez] = get_data("efield.txt",
                                                        nskip=1, dims=[6,None])
        [_,_,_,Hx,Hy,Hz] = get_data("hfield.txt",nskip=1,dims=[6,None])

        self.Bx = Hx*picmi.mu0
        self.By = Hy*picmi.mu0
        self.Bz = Hz*picmi.mu0

        # Interpolate them at cell centers (as prescribed by Warp doc)
        self.d = abs(self.x[1]-self.x[0])
        # Number of mesh cells
        self.NNx = int(round(2*np.max(self.x)/self.d))
        self.NNy = int(round(2*np.max(self.y)/self.d))
        self.NNz = int(round(2*np.max(self.z)/self.d))
        # Number of mesh vertices
        self.nnx = self.NNx + 1
        self.nny = self.NNy + 1
        self.nnz = self.NNz + 1

        self.Ex3d = self.Ex.reshape(self.nnz, self.nny, 
                                    self.nnx).transpose(2, 1, 0)
        self.Ey3d = self.Ey.reshape(self.nnz, self.nny, 
                                    self.nnx).transpose(2, 1, 0)
        self.Ez3d = self.Ez.reshape(self.nnz, self.nny, 
                                    self.nnx).transpose(2, 1, 0)
        self.Bx3d = self.Bx.reshape(self.nnz, self.nny, 
                                    self.nnx).transpose(2, 1, 0)
        self.By3d = self.By.reshape(self.nnz, self.nny,
                                    self.nnx).transpose(2, 1, 0)
        self.Bz3d = self.Bz.reshape(self.nnz, self.nny, 
                                    self.nnx).transpose(2, 1, 0)
        self.x3d = self.x.reshape(self.nnz, self.nny, 
                                  self.nnx).transpose(2, 1, 0)
        self.y3d = self.y.reshape(self.nnz, self.nny, 
                                  self.nnx).transpose(2, 1, 0)
        self.z3d = self.z.reshape(self.nnz, self.nny, 
                                  self.nnx).transpose(2, 1, 0)

        # Rescale the fields at convenience
        maxE = max_rescale
        kk = maxE/np.max(abs(self.Ey3d))

        self.Ex3d *= kk
        self.Ey3d *= kk
        self.Ez3d *= kk
        self.Bx3d *= kk
        self.By3d *= kk
        self.Bz3d *= kk

        self.Exx = 0.125*(self.Ex3d[0:-1, 0:-1, 0:-1]
                        + self.Ex3d[0:-1:, 0:-1, 1:]
                        + self.Ex3d[0:-1, 1:, 0:-1]
                        + self.Ex3d[0:-1, 1:, 1:]
                        + self.Ex3d[1:, 1:, 0:-1]
                        + self.Ex3d[1:, 1:, 1:]
                        + self.Ex3d[1:, 0:-1, 1:]
                        + self.Ex3d[1:, 0:-1, 0:-1])

        self.Eyy = 0.125*(self.Ey3d[0:-1, 0:-1, 0:-1]
                        + self.Ey3d[0:-1:, 0:-1, 1:]
                        + self.Ey3d[0:-1, 1:, 0:-1]
                        + self.Ey3d[0:-1, 1:, 1:]
                        + self.Ey3d[1:, 1:, 0:-1]
                        + self.Ey3d[1:, 1:, 1:]
                        + self.Ey3d[1:, 0:-1, 1:]
                        + self.Ey3d[1:, 0:-1, 0:-1])

        self.Ezz = 0.125*(self.Ez3d[0:-1, 0:-1, 0:-1]
                        + self.Ez3d[0:-1:, 0:-1, 1:]
                        + self.Ez3d[0:-1, 1:, 0:-1]
                        + self.Ez3d[0:-1, 1:, 1:]
                        + self.Ez3d[1:, 1:, 0:-1]
                        + self.Ez3d[1:, 1:, 1:]
                        + self.Ez3d[1:, 0:-1, 1:]
                        + self.Ez3d[1:, 0:-1, 0:-1])

        self.Bxx = 0.125*(self.Bx3d[0:-1, 0:-1, 0:-1]
                        + self.Bx3d[0:-1:, 0:-1, 1:]
                        + self.Bx3d[0:-1, 1:, 0:-1]
                        + self.Bx3d[0:-1, 1:, 1:]
                        + self.Bx3d[1:, 1:, 0:-1]
                        + self.Bx3d[1:, 1:, 1:]
                        + self.Bx3d[1:, 0:-1, 1:]
                        + self.Bx3d[1:, 0:-1, 0:-1])

        self.Byy = 0.125*(self.By3d[0:-1, 0:-1, 0:-1]
                        + self.By3d[0:-1: , 0:-1, 1:]
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

        # Lattice spatial parameters
        self.zs = np.min(self.z3d) - self.d/2.
        self.ze = np.max(self.z3d) + self.d/2.
        self.xs = np.min(self.x3d) - self.d/2.
        self.ys = np.min(self.y3d)-self.d/2.
        # Lattice temporal parameters
        self.Tf = 25e-9
        self.freq = 400*1e6
        self.Nt = 1000
        self.phase_disp=0
        self.time_array = np.linspace(0., self.Tf, self.Nt)
        self.data_arrayE = np.sin(self.time_array*self.freq*2*np.pi 
                                + self.phase_disp)
        self.data_arrayB = np.sin(self.time_array*self.freq*2*np.pi 
                                - np.pi/2 + self.phase_disp)

        # Create overlapped lattice elements to have E and B in the same region
        self.ie, self.egrid = picmi.warp.addnewegrd(self.zs, self.ze,
                                                    dx = self.d, dy = self.d,
                                                    xs = self.xs, ys = self.ys,
                                                    time = self.time_array,
                                                    data = self.data_arrayE,
                                                    ex = self.Ex3d, 
                                                    ey = self.Ey3d,
                                                    ez = self.Ez3d)


        picmi.warp.addnewbgrd(self.zs, self.ze, dx = self.d, dy = self.d,
                              xs = self.xs, ys = self.ys,
                              time = self.time_array, data = self.data_arrayB,
                              bx = self.Bx3d, by = self.By3d, bz = self.Bz3d)
