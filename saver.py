import numpy as np
import scipy.io as sio
import os

class Saver:

    def __init__(self, flag_output, flag_checkpointing, tot_nsteps, n_bunches,
                 nbins, output_filename= None, temps_filename = None):
        self.flag_checkpointing = flag_checkpointing
        self.flag_output = flag_output
        self.temps_filename = temps_filename
        self.tot_nsteps = tot_nsteps
        self.n_bunches = n_bunches
        self.nbins = nbins
        self.output_filename = output_filename

        if (self.flag_output and not
           (self.flag_checkpointing and os.path.exists(self.temps_filename))):
            self.init_empty_outputs()
        else:
             self.restore_outputs_from_file()

    def init_empty_outputs(self):
        self.numelecs = np.zeros(self.tot_nsteps)
        self.numelecs_tot = np.zeros(self.tot_nsteps)
        self.N_mp = np.zeros(self.tot_nsteps)
        self.xhist = np.zeros((self.n_bunches,self.nbins))
        self.bins = np.zeros(self.nbins)

    def restore_outputs_from_file(self):
        dict_init_dist = sio.loadmat(self.temps_filename)
        if self.flag_output:
            self.numelecs = dict_init_dist['numelecs'][0]
            self.N_mp = dict_init_dist['N_mp'][0]
            self.numelecs_tot = dict_init_dist['numelecs_tot'][0]
            self.xhist = dict_init_dist['xhist']
            self.bins = dict_init_dist['bins']
            self.b_pass = dict_init_dist['b_pass']
    
    def save_checkpoint(self, b_pass, elecbw, secelecw):
        dict_out_temp = {}
        print('Saving a checkpoint!')
        dict_out_temp['x_mp'] = np.concatenate((secelecw.getx(),
                                                elecbw.getx()))
        dict_out_temp['y_mp'] = np.concatenate((secelecw.gety(),
                                                elecbw.gety()))
        dict_out_temp['z_mp'] = np.concatenate((secelecw.getz(),
                                                elecbw.gety()))
        dict_out_temp['vx_mp'] = np.concatenate((secelecw.getvx(),
                                                 elecbw.getvx()))
        dict_out_temp['vy_mp'] = np.concatenate((secelecw.getvy(),
                                                 elecbw.getvy()))
        dict_out_temp['vz_mp'] = np.concatenate((secelecw.getvz(),
                                                 elecbw.getvz()))
        dict_out_temp['nel_mp'] = np.concatenate((secelecw.getw(),
                                                  elecbw.getw()))
        if self.flag_output:
            dict_out_temp['numelecs'] = self.numelecs
            dict_out_temp['numelecs_tot'] = self.numelecs_tot
            dict_out_temp['N_mp'] = self.N_mp
            dict_out_temp['xhist'] = self.xhist
            dict_out_temp['bins'] = self.bins

        dict_out_temp['b_pass'] = b_pass

        filename = 'temp_mps_info.mat'

        sio.savemat(filename, dict_out_temp)

    def update_outputs(self, sw, ew, nz, n_step):
        secelec_w = sw.getw()
        elecb_w = ew.getw()
        elecs_density = (sw.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)] 
                       + ew.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)])
        elecs_density_tot = (sw.get_density(l_dividebyvolume=0)[:,:,:] 
                           + ew.get_density(l_dividebyvolume=0)[:,:,:])
        self.numelecs[n_step] = np.sum(elecs_density)
        self.numelecs_tot[n_step] = np.sum(elecs_density_tot)
        self.N_mp[n_step] = len(secelec_w)+len(elecb_w)

    def dump_outputs(self, xmin, xmax, secelecw, b_pass):
        dict_out = {}
        dict_out['numelecs'] = self.numelecs
        dict_out['numelecs_tot'] = self.numelecs_tot
        dict_out['N_mp'] = self.N_mp
        # Compute the x-position histogram
        (self.xhist[b_pass-1], self.bins) = np.histogram(secelecw.getx(), 
                                                     range = (xmin,xmax), 
                                                     bins = self.nbins, 
                                                     weights = secelecw.getw(), 
                                                     density = False)
        dict_out['bins'] = self.bins
        dict_out['xhist'] = self.xhist
        sio.savemat(self.output_filename, dict_out)
