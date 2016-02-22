import os, sys
import numpy as np
import scipy.interpolate
import scipy.optimize
import astropy.io.fits as fitsio
from astropy.wcs import WCS

from tools import *
from plotting import *

class WISP_Catalog():

    def __init__(self,par_num,grism,data_dir):

        self.data_dir   = data_dir

        self.par_num  = par_num
        self.grism    = grism
        self.direct   = direct_key[self.grism]

        self.par_dir    = os.path.join(self.data_dir,'Par%i' % self.par_num)
        self.direct_dir = os.path.join(self.par_dir,'DATA/DIRECT_GRISM')

        self.set_catalog()

    def set_catalog(self):

        self._catalog = np.genfromtxt('%s/fin_%s.cat' % (self.direct_dir,self.direct), dtype=[('RA_DEC_NAME','>S25'),('NUMBER',int),('X_IMAGE',float),('Y_IMAGE',float),('A_IMAGE',float),('B_IMAGE',float),('THETA_IMAGE',float),('X_WORLD',float),('Y_WORLD',float),('A_WORLD',float),('B_WORLD',float),('THETA_WORLD',float),('MAG',float),('MAGERR_AUTO',float),('CLASS_STAR',float),('FLAGS',int)],)

        bbox_x, bbox_y = bboxEllipse(self._catalog['A_IMAGE'],self._catalog['B_IMAGE'],self._catalog['THETA_IMAGE'])

        dtyp = [('PAR_NUM',int),('GRISM','S4')] + self._catalog.dtype.descr + [('BBOX_X',float),('BBOX_Y',float)]
        self.catalog = np.recarray(len(self._catalog),dtype=dtyp)
        for x in self._catalog.dtype.names: self.catalog[x] = self._catalog[x]
        self.catalog['PAR_NUM'] = self.par_num
        self.catalog['GRISM']   = self.grism
        self.catalog['BBOX_X']  = 2*bbox_x
        self.catalog['BBOX_Y']  = 2*bbox_y

    def get_catalog(self):
        return self.catalog

class WISP_Field():

    def __init__(self,data_dir,output_dir,catalog=None,par_num=None,grism=None,background=None):

        self.data_dir = data_dir
        self.output_dir = output_dir
        os.system('mkdir -p %s/stamps' % self.output_dir)

        if catalog is not None:
            self.catalog = catalog
        else:
            self.catalog = WISP_Catalog(par_num=par_num,grism=grism,data_dir=data_dir).get_catalog()

        self.par_num    = self.catalog['PAR_NUM'][0]
        self.grism      = self.catalog['GRISM'][0]
        self.direct     = direct_key[self.grism]
        self.background = background

        self.par_dir    = os.path.join(self.data_dir,'Par%i' % self.par_num)
        self.direct_dir = os.path.join(self.par_dir,'DATA/DIRECT_GRISM')

    def process(self):
        self.get_direct()
        self.get_background(self.background)
        self.mk_stamps()

    def get_direct(self):
        self.direct_hdr = fitsio.getheader('%s/%sW_sci.fits' % (self.direct_dir,self.direct))
        self.direct_img = fitsio.getdata('%s/%sW_sci.fits' % (self.direct_dir,self.direct))
        self.direct_wcs = WCS(self.direct_hdr)

    def get_background(self,background):

        if not background:
            masked = mask_direct_image(self.direct_img, self.direct_hdr, self.catalog, fill_value=-99.)
            self.sky_median = np.median(masked[masked != -99.])
        else:
            print "Using user-specified background: ", background
            self.sky_median = background
        self.direct_img_sub = self.direct_img.copy() - self.sky_median

    def get_direct_contams(self,entry):

        dx, dy = 5*entry['BBOX_X'], 5*entry['BBOX_Y']
        return get_direct_contams(self.catalog,entry,dx,dy)

    def mk_stamps(self):

        for entry in self.catalog:
            sys.stdout.write("\rMaking direct image stamp for Obj#%i out of %i ... " % (entry['NUMBER'],len(self.catalog)),)
            sys.stdout.flush()
            contams = self.get_direct_contams(entry)
            masked  = mask_direct_image(self.direct_img_sub, self.direct_hdr, contams, fill_value=self.sky_median)
            stamp, stphdr = cut_stamps(masked,entry,self.direct_wcs,self.direct_hdr,scale=5.)
            fitsio.writeto('%s/stamps/stamp_%s_%i.fits' % (self.output_dir,self.direct,entry['NUMBER']), data=stamp, header=stphdr, clobber=True)
        sys.stdout.write("done.\n")

class WISP_Source():

    def __init__(self,catalog,entry,data_dir,output_dir,s=5):

        self.data_dir = data_dir
        self.output_dir = output_dir
        os.system('mkdir -p %s/plots' % self.output_dir)
        os.system('mkdir -p %s/clean' % self.output_dir)

        self.catalog  = catalog
        self.entry    = entry
        self.grism    = self.entry['GRISM']
        self.obj_num  = self.entry['NUMBER']
        self.par_num  = self.entry['PAR_NUM']
        self.direct   = direct_key[self.grism]
        self.s        = s

        if self.s % 2 == 0:
            raise Exception('Provide an odd number of pixels to smooth over.')

        self.par_dir    = os.path.join(self.data_dir,'Par%i' % self.par_num)
        self.grism_dir  = os.path.join(self.par_dir,'%s_DRIZZLE' % self.grism)

        self.px_scale_grism = 0.128254
        self.px_scale_direct = 0.08
        self.px_scale_cor = self.px_scale_direct / self.px_scale_grism

        self.get_grism_sens()
        self.get_grism()
        self.get_grism_contams()
        self.get_grism1D()
        self.get_profiles()
        self.get_prior()

    def get_grism_sens(self,fill_value=1e-15):

        sens_file = fitsio.getdata('WFC3.%s.1st.sens.fits' % self.grism)
        waves, sens = sens_file['WAVELENGTH'], sens_file['SENSITIVITY']
        sens[sens==0] = fill_value
        self.grism_sens = scipy.interpolate.interp1d(waves,sens,bounds_error=False,fill_value=fill_value)

    def get_grism(self):

        self.grism_hdr = fitsio.getheader('%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'SCI')
        self.grism_img = fitsio.getdata(  '%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'SCI')
        self.grism_err = fitsio.getdata(  '%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'ERR')
        self.grism_len = self.grism_hdr['NAXIS1']

        self.grism_mask = self.grism_err==0
        self.grism_img  = np.ma.masked_array(self.grism_img,mask=self.grism_mask,fill_value=np.NaN)
        self.grism_err  = np.ma.masked_array(self.grism_err,mask=self.grism_mask,fill_value=np.NaN)

        self.grism_model = self.grism_img.copy() * 0
        self.grism_img_s = self.grism_img.copy() * 0
        self.grism_err_s = self.grism_img.copy() * 0

        n = (self.s-1.)/2.
        self.grism_img_s[:,n:-n] =            moving_average(self.grism_img,    s=self.s)
        self.grism_err_s[:,n:-n] = np.ma.sqrt(moving_average(self.grism_err**2, s=self.s) / self.s)

    def get_grism1D(self):

        self.grism1D_img  =            np.ma.average(self.grism_img   ,axis=-1)
        self.grism1D_err  = np.ma.sqrt(np.ma.average(self.grism_err**2,axis=-1) / self.grism_err.shape[0])
        self.grism1D_mask = self.grism1D_img.mask

        self.x  = np.arange(len(self.grism1D_img))
        grism1D_img_interpfn = scipy.interpolate.interp1d(self.x[~self.grism1D_mask],self.grism1D_img[~self.grism1D_mask],bounds_error=False,fill_value=0)
        grism1D_err_interpfn = scipy.interpolate.interp1d(self.x[~self.grism1D_mask],self.grism1D_err[~self.grism1D_mask],bounds_error=False,fill_value=0)

        self.xf = np.linspace(min(self.x),max(self.x),len(self.x)*100.)
        self.grism1D_img = grism1D_img_interpfn(self.xf)
        self.grism1D_err = grism1D_err_interpfn(self.xf)
        self.grism1D_img = np.ma.masked_array(self.grism1D_img,mask=self.grism1D_err==0,fill_value=np.NaN)
        self.grism1D_err = np.ma.masked_array(self.grism1D_err,mask=self.grism1D_err==0,fill_value=np.NaN)

    def get_grism_contams(self):
        dy, dx = np.asarray(self.grism_img.shape)
        dy = dy / self.px_scale_cor
        self.contams = get_grism_contams(self.catalog,self.entry,dx,dy)

    def get_subpx_pars(self):
        return self.subpx_pars

    def get_profiles(self):
        self.d_prof =  mk_stamp_profile(self.entry,self.px_scale_cor,self.output_dir,self.direct)
        self.c_prof = [mk_stamp_profile(contam,self.px_scale_cor,self.output_dir,self.direct) for contam in self.contams]
        self.d_cen  = self.grism_img.shape[0] / 2.
        self.c_cen  = (self.contams['Y_IMAGE'] - self.entry['Y_IMAGE']) * self.px_scale_cor + self.d_cen
        self.model_args = (self.d_prof, self.d_cen, self.c_prof, self.c_cen)

    def lnl(self,pars,x,ydata,yerr):
        model_ev = profile_model(x,pars,self.model_args)
        invsig2  = np.ma.power(yerr,-1)
        do_sum   = (ydata-model_ev)**2 * invsig2 - np.ma.log(invsig2)
        lnl      = 0.5 * np.ma.sum(do_sum.filled(1e5))
        return lnl

    def get_subpx_shifts(self):
        x0 = [0,] + [0.01,0] * (len(self.contams)+1)
        bounds = [(-5,5),] + [(0,5),(-2,2)] * (len(self.contams)+1)
        self.subpx_pars = scipy.optimize.minimize(self.lnl,x0=x0,bounds=bounds,args=(self.xf,self.grism1D_img,self.grism1D_err))['x']
        print "Sub-pixelling  done for Obj #%i" % self.entry['NUMBER']

    def get_prior(self):
        waves = get_waves(np.arange(self.grism_len), self.grism_hdr)
        self.prior = np.zeros((len(self.contams)+1,len(waves)))
        self.prior[0,:] = self.grism_sens(waves)
        for i,contam in enumerate(self.contams):
            del_x = (contam['X_IMAGE'] - self.entry['X_IMAGE']) * self.px_scale_cor
            waves = get_waves(np.arange(self.grism_len) - del_x, self.grism_hdr)
            self.prior[i+1,:] = self.grism_sens(waves)

    def lnl2(self,pars_,x,ydata,yerr,prior):
        pars = self.subpx_pars.copy()
        pars[1::2] = pars_
        lnl = self.lnl(pars,x,ydata,yerr)
        p = np.ma.sum(np.ma.log(prior[1:]/prior[0]))
        return lnl - p

    def clean_grism(self):
        x0 = self.subpx_pars[1::2]
        bounds = [(0,5),] * len(x0)
        for i in range(self.grism_len):
            ydata, yerr = self.grism_img_s[:,i], self.grism_err_s[:,i]
            if not all(yerr==0.) and any(np.isfinite(ydata)):
                pars_ = scipy.optimize.minimize(self.lnl2,x0=x0,bounds=bounds,args=(self.x,ydata,yerr,self.prior[:,i]))['x']
                pars = self.subpx_pars.copy()
                pars[1::2] = pars_
                self.grism_model[:,i] = contam_model(self.x,pars,self.c_prof,self.c_cen)
        print "Grism cleaning done for Obj #%i" % self.entry['NUMBER']
        self.grism_img_c = self.grism_img - self.grism_model

    def save_fits(self):
        hdulist = fitsio.HDUList([])
        for data,extname in zip([self.grism_img,self.grism_err,self.grism_model,self.grism_img_c],
                                    ['SCI','ERR','MODEL','CLEAN']):
            data = data.filled(fill_value=np.NaN)
            hdu = fitsio.PrimaryHDU(header=self.grism_hdr,data=data)
            hdu.name = extname
            hdulist.append(hdu)
        hdulist.writeto(self.output_dir+'/clean/aXeWFC3_%s_clean_ID%s.fits' % (self.grism,self.obj_num),clobber=True)

    def process(self):

        self.get_subpx_shifts()
        self.clean_grism()
        self.save_fits()

        #plot_prior(self.prior,self.grism_hdr)
        plot_subpx_shifts(self.grism,self.entry,self.subpx_pars,self.xf,self.grism1D_img,self.model_args,self.output_dir)
        plot_clean_grism(self.grism,self.entry,self.grism_hdr,self.grism_img,self.grism_model,self.grism_img_c,
                         self.contams,self.subpx_pars[2::2],self.px_scale_cor,self.output_dir)
