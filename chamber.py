from warp import picmi
import numpy as np

class RectChamber:
    
    def __init__(self, width, height, z_start, z_end, ghost_x = 1e-3, 
                 ghost_y = 1e-3, ghost_z = 1e-3, condid = 1):

        print('Using rectangular chamber with xaper: %1.2e, yaper: %1.2e' 
                                                        %(width/2., height/2.))
        self.width = width
        self.height = height
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -width/2 - self.ghost_x
        self.xmax = -self.xmin
        self.ymin = -height/2 - self.ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - self.ghost_z
        self.zmax = z_end + self.ghost_z

        self.lower_bound = [-width/2, -height/2, z_start]
        self.upper_bound = [width/2, height/2, z_end]

        self.condid = condid

        upper_box = picmi.warp.YPlane(y0 = height/2, ysign = 1,
                                      condid = condid)
        lower_box = picmi.warp.YPlane(y0 = -height/2, ysign = -1, 
                                      condid = condid)
        left_box = picmi.warp.XPlane(x0 = width/2, xsign = 1, condid = condid)
        right_box = picmi.warp.XPlane(x0 = -width/2, xsign =-1,
                                      condid = condid)
        
        self.conductors = upper_box + lower_box + left_box + right_box

    def is_outside(self, xx, yy, zz):
        width = self.width
        height = self.height
        z_start = self.z_start
        z_end = self.z_end
        return np.logical_or.reduce([abs(xx) > width/2, abs(yy) > height/2, 
                                     zz < z_start, zz > z_end])

class LHCChamber:
    
    def __init__(self, length, z_start, z_end, ghost_x = 1e-3, ghost_y = 1e-3,
                 ghost_z = 1e-3, condid = 1):
        
        print('Using the LHC chamber')

        self.height = 36e-3
        self.radius = 23e-3
        self.width = self.radius*2
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -width/2. - ghost_x
        self.xmax = -self.xmin
        self.ymin = -height/2 - ghost_y
        self.ymax = -self.ymin
        self.zmin = zs_dipo - ghost_z
        self.zmax = ze_dipo + ghost_z

        self.lower_bound = [-radius, -radius, z_start]
        self.upper_bound = [radius, radius, z_end]

        upper_box = picmi.warp.YPlane(y0 = height/2, ysign = 1,
                                      condid = condid)
        lower_box = picmi.warp.YPlane(y0 = -height/2, ysign = -1,
                                      condid = condid)
        pipe = picmi.warp.ZCylinderOut(radius = radius,  length = self.length, 
                                       condid = condid)

        self.conductors = pipe + upper_box + lower_box

    def is_outside(self, xx, yy, zz):
        r0_sq = np.square(x0) + np.square(y0)
        return np.logical_or.reduce([r0sq > self.radius**2, 
                                         abs(yy) > self.height,
                                         zz < z_start, zz > z_end])

class CircChamber:

    def __init__(self, radius, z_start, z_end, ghost_x = 1e-3, ghost_y = 1e-3,
                 ghost_z = 1e-3, condid = 1):

        print('Using a circular chamber with radius %1.2e' %radius)

        self.radius = radius      
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -radius - ghost_x
        self.xmax = -self.xmin
        self.ymin = -radius - ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - ghost_z
        self.zmax = z_end + ghost_z

        self.lower_bound = [-radius, -radius, z_start]
        self.upper_bound = [radius, radius, z_end]

        pipe = picmi.warp.ZCylinderOut(radius = radius, length = self.length, 
                                       condid = condid)

        self.conductors = pipe

    def is_outside(self, xx, yy, zz):
        r0_sq = np.square(x0) + np.square(y0)
        return np.logical_or.reduce([r0sq > self.radius**2,
                                     zz < z_start, zz > z_end])

class CrabCavity:

    def __init__(self, z_start, z_end, ghost_x = 1e-3, ghost_y = 1e-3,
                 ghost_z = 1e-3, condid = 1):

        print('Simulating ECLOUD in a crab cavity')

        self.z_start = z_start
        self.z_end = z_end

        self.l_main_y = 242e-3
        self.l_main_x = 300e-3
        self.l_main_z = 350e-3
        self.l_beam_pipe = 84e-3
        self.l_int = 62e-3
        self.l_main_int_y = self.l_main_y - self.l_beam_pipe/2
        self.l_main_int_z = self.l_main_z/2 - self.l_int
        self.l_main_int_x = self.l_main_x/2 - self.l_int
        # chamber_are makes sense just when we can compare to 

        assert z_start < - self.l_main_z/2, 'z_start must be lower than -175mm'
        assert z_end > self.l_main_z/2, 'z_end must be higher than 175mm'

        self.xmin = -self.l_main_x/2 - self.ghost_x
        self.xmax = -self.xmin
        self.ymin = -self.l_main_y/2 - ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - ghost_z
        self.zmax = z_end + ghost_z

        box1 = picmi.warp.Box(zsize = self.zmax - self.zmin,
                              xsize = self.xmax - self.xmin,
                              ysize = self.ymax - self.ymin, condid = condid)
        box2 = picmi.warp.Box(zsize = self.zmax - self.zmin,
                              xsize = l_beam_pipe, 
                              ysize = l_beam_pipe, condid = condid)
        box3 = picmi.warp.Box(zsize = l_main_z,
                              xsize = l_main_x,
                              ysize=l_main_y, condid = condid)
        ycen1 = l_beam_pipe/2 + l_main_int_y
        ycen2 = -ycen1
        box4 = picmi.warp.Box(zsize = 2*self.l_main_int_z, 
                              xsize = 2*self.l_main_int_x, 
                              ysize = 2*self.l_main_int_y, ycent=ycen1, 
                              condid = condid)
        box5 = picmi.warp.Box(zsize = 2*self.l_main_int_z, 
                              xsize = 2*self.l_main_int_x, 
                              ysize = 2*self.l_main_int_y, ycent=ycen2, 
                              condid = condid)

        self.conductors = box1 - box2 - box3 + box4 + box5

    def is_outside(self, xx,yy,zz):
        flag_out_box = np.logical_and.reduce([abs(xx) > self.l_main_x/2, 
                                              abs(yy) > self.l_main_y/2,
                                              abs(zz) > self.l_main_z/2])

        flag_out_poles = np.logical_and.reduce([abs(xx) < self.l_main_int_x,
                                                abs(zz) < self.l_main_int_z,
                                                abs(yy) > self.l_beam_pipe/2])
        zs_pipe_left = self.z_start
        ze_pipe_left = -self.l_main_z/2
        zs_pipe_right = self.l_main_z/2
        ze_pipe_right = self.z_end
        flag_out_pipe_l = np.logical_and.reduce([ abs(xx) > self.l_beam_pipe,
                                                  abs(yy) > self.l_beam_pipe,
                                                  zz < zs_pipe_left,
                                                  zz > ze_pipe_left])
        flag_out_pipe_r = np.logical_and.reduce([ abs(xx) > self.l_beam_pipe,
                                                  abs(yy) > self.l_beam_pipe,
                                                  zz < zs_pipe_right,
                                                  zz > ze_pipe_right])


        flag_out_left = np.logical_and(flag_out_box,flag_out_pipe_left)
        flag_out_right = np.logical_and(flag_out_box, flag_out_pipe_left)
        
        return np.logical_or.reduce([flag_out_box, flag_out_poles, 
                                     flag_out_left, flag_out_right])
    

