import os, sys
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.stats

import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat

import iminuit

import astropy.io.fits as fitsio
from astropy.wcs import WCS

from tools import *
from tools import direct_key
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

class WISP_Source():

    def __init__(self,par_num,obj_num,grism,data_dir,output_dir,background=None,s=5,config_dir='config'):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config_dir = config_dir
        os.system('mkdir -p %s/Par%i/plot_grism' % (self.output_dir,par_num))
        os.system('mkdir -p %s/Par%i/plot_prof'  % (self.output_dir,par_num))
        os.system('mkdir -p %s/Par%i/clean'      % (self.output_dir,par_num))
        os.system('mkdir -p %s/Par%i/stamps'     % (self.output_dir,par_num))

        self.catalog    = WISP_Catalog(par_num=par_num,grism=grism,data_dir=data_dir).get_catalog()

        entry = self.catalog[self.catalog['NUMBER'] == obj_num]
        if len(entry) == 1: self.entry = entry[0]
        elif len(entry) == 0: raise Exception("No object with ID#%i found in Par%i." % (obj_num,par_num))
        else: raise Exception("Multiple objects with ID#%i found in Par%i." % (obj_num,par_num))

        self.grism      = self.entry['GRISM'].decode("utf-8")
        self.obj_num    = self.entry['NUMBER']
        self.par_num    = self.entry['PAR_NUM']
        self.direct     = direct_key[self.grism]
        self.background = background
        self.s          = s

        if self.s % 2 == 0:
            raise Exception('Provide an odd number of pixels to smooth over.')

        self.par_dir    = os.path.join(self.data_dir,'Par%i' % self.par_num)
        self.direct_dir = os.path.join(self.par_dir,'DATA/DIRECT_GRISM')
        self.grism_dir  = os.path.join(self.par_dir,'%s_DRIZZLE' % self.grism)

        self.px_scale_grism = 0.128254
        self.px_scale_direct = 0.08
        self.px_scale_cor = self.px_scale_direct / self.px_scale_grism

        self.likelihood_space = []
        self.profile_uncertainties = []
        self.last_pars = None

        self.get_direct()
        self.get_background(self.background)
        self.get_grism_sens()
        self.get_grism()
        self.get_grism_contams()
        self.get_grism1D()
        self.get_profiles()
        self.get_prior()

    def get_direct(self):

        self.direct_hdr = fitsio.getheader('%s/%sW_sci.fits' % (self.direct_dir,self.direct))
        self.direct_img = fitsio.getdata('%s/%sW_sci.fits' % (self.direct_dir,self.direct))
        self.direct_wcs = WCS(self.direct_hdr)

    def get_background(self,background):

        if not background:
            masked = mask_direct_image(self.direct_img, self.direct_hdr, self.catalog, fill_value=-99.)
            self.sky_median = np.median(masked[masked != -99.])
        else:
            print ("Using user-specified background: ", background)
            self.sky_median = background
        self.direct_img_sub = self.direct_img.copy() - self.sky_median

    def get_direct_contams(self,entry):

        dx, dy = 5*entry['BBOX_X'], 5*entry['BBOX_Y']
        return get_direct_contams(self.catalog,entry,dx,dy)

    def mk_stamps(self,entry):

        if not os.path.isfile('%s/Par%i/stamps/stamp_%s_%i.fits' % (self.output_dir,entry['PAR_NUM'],self.direct,entry['NUMBER'])):
            contams = self.get_direct_contams(entry)
            masked  = mask_direct_image(self.direct_img_sub, self.direct_hdr, contams, fill_value=self.sky_median)
            stamp, stphdr = cut_stamps(masked,entry,self.direct_wcs,self.direct_hdr,scale=5.)
            fitsio.writeto('%s/Par%i/stamps/stamp_%s_%i.fits' % (self.output_dir,entry['PAR_NUM'],self.direct,entry['NUMBER']), data=stamp, header=stphdr, overwrite=True)

    def get_grism_sens(self,fill_value=1e-15):

        sens_file = fitsio.getdata('{}/WFC3.{}.1st.sens.fits'.format(self.config_dir, self.grism))
        waves, sens = sens_file['WAVELENGTH'], sens_file['SENSITIVITY']
        sens[sens==0] = fill_value
        self.grism_sens = scipy.interpolate.interp1d(waves,sens,bounds_error=False,fill_value=fill_value)

    def get_grism_original_fits(self) :
        return fitsio.open('%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num))

    def get_grism(self):

        self.grism_file = '%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num)
        self.grism_hdr = fitsio.getheader('%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'SCI')
        self.grism_img = fitsio.getdata(  '%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'SCI')
        self.grism_err = fitsio.getdata(  '%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'ERR')
        self.grism_con = fitsio.getdata(  '%s/aXeWFC3_%s_mef_ID%i.fits' % (self.grism_dir,self.grism,self.obj_num),'CON')
        self.grism_len = self.grism_hdr['NAXIS1']

        self.grism_mask = self.grism_err==0
        self.grism_img  = np.ma.masked_array(self.grism_img,mask=self.grism_mask,fill_value=np.NaN)
        self.grism_err  = np.ma.masked_array(self.grism_err,mask=self.grism_mask,fill_value=np.NaN)

        self.grism_model = self.grism_img.copy() * 0
        self.grism_img_s = self.grism_img.copy() * 0
        self.grism_err_s = self.grism_img.copy() * 0
        self.uncertain_grism_model = unumpy.uarray(self.grism_model, self.grism_model)

        n = int((self.s-1.)/2.)
        self.grism_img_s[:,n:-n] =            moving_average(self.grism_img,    s=self.s)
        self.grism_err_s[:,n:-n] = np.ma.sqrt(moving_average(self.grism_err**2, s=self.s) / self.s)

        self.uncertain_grism_img = unumpy.uarray(self.grism_img, self.grism_err)
        self.uncertain_grism_img_s = unumpy.uarray(self.grism_img_s, self.grism_err_s)


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
        # save model normalizations and corresponding covariances for all contaminants
        # and the overall grism zero-point offset
        self.grism_contam_model_pars = np.zeros(shape=(self.grism_img.shape[1], len(self.contams)+1))
        self.grism_contam_model_covmats = np.zeros(shape=(self.grism_img.shape[1], *([len(self.contams)+1]*2)))

    def get_subpx_pars(self):
        return self.subpx_pars

    def get_profiles(self):

        _ = [self.mk_stamps(self.entry),] + [self.mk_stamps(contam) for contam in self.contams]
        self.d_prof =  mk_stamp_profile(self.entry,self.px_scale_cor,self.output_dir,self.direct)
        self.c_prof = [mk_stamp_profile(contam,self.px_scale_cor,self.output_dir,self.direct) for contam in self.contams]
        self.d_cen  = self.grism_img.shape[0] / 2.
        self.c_cen  = (self.contams['Y_IMAGE'] - self.entry['Y_IMAGE']) * self.px_scale_cor + self.d_cen
        self.model_args = (self.d_prof, self.d_cen, self.c_prof, self.c_cen)

    def lnl(self,pars,x,ydata,yerr):
        model_ev = profile_model(x,pars,self.model_args)
        invsig2  = np.ma.power(yerr,-1) # FIXME: Is this a bug?
        do_sum   = -0.5*(((ydata-model_ev)*invsig2)**2 + np.log(2*np.pi)) + np.ma.log(invsig2)
        lnl      = np.ma.sum(do_sum.filled(1e5))

        return -lnl

    def get_subpx_shifts(self):
        """
        This method accounts for sub-pixel offsets in the cross-dispersion direction between the
        contamination models (derived as interpolants of projected direct images) and the dispersed
        contamination as measured by the grism.

        It performs a constrained global minimization for all contaminating sources and the
        target source which allows the model for each contaminant and the target to displace
        slighly in the cross-dispersion direction.

        The contamination and target models are evaluated on a fine (factor 100 supersampled)
        grid in the cross-dispersion direction.
        """
        # The zeroth element is the global offset (zero-point) for the whole grism image.
        # Successive pairs of elements are the normalizations and cross-dispersion offsets
        # for each of the contamination models and the target.
        x0 = [0,] + [0.01,0] * (len(self.contams)+1)
        bounds = [(-5,5),] + [(0,5),(-2,2)] * (len(self.contams)+1)
        self.subpx_pars = scipy.optimize.minimize(self.lnl,x0=x0,bounds=bounds,args=(self.xf,self.grism1D_img,self.grism1D_err))['x']
        print ("Sub-pixelling  done for Obj #%i" % self.entry['NUMBER'])

    def get_prior(self):
        waves = get_waves(np.arange(self.grism_len), self.grism_hdr)
        self.prior = np.zeros((len(self.contams)+1,len(waves)))
        self.prior[0,:] = self.grism_sens(waves)
        for i,contam in enumerate(self.contams):
            # populate subsequent rows with the sensitivity curve offset in wavelength
            # according to the expected position of the first order diffraction in
            # the grism.
            del_x = (contam['X_IMAGE'] - self.entry['X_IMAGE']) * self.px_scale_cor
            waves = get_waves(np.arange(self.grism_len) - del_x, self.grism_hdr)
            self.prior[i+1,:] = self.grism_sens(waves)

    def lnl2(self,pars_,x,ydata,yerr,prior):
        # Retrieve best fitting parameters derived from subpixelling - only the subset
        # Corresponding to the normalizations of the target and contamination models
        # (supplied as _pars) should be varied so...
        pars = self.subpx_pars.copy()
        # Update the subpixelling best fit parameters with the current trial normalization
        # values...
        pars[1::2] = pars_
        # Now compute the likelihood of the model given those parameters.
        lnl = self.lnl(pars,x,ydata,yerr)
        # One can estimate the prior expectation of flux level for the column based on the
        # grism sensitivity curve and the expected positions of the 1st order dispersion
        # flux for each contaminant and the target.
        #
        # ** The units of the sensitivity curve are e−/s per erg cm^{−2} s^{−1} Ang.^{−1}
        # ** The prior describes the superposed sensitivity curve values for each pixel in a
        # single image column.
        # ** The zeroth element of the prior original unshifted sensitivity curve.
        p = np.ma.sum(np.ma.log(prior[1:]/prior[0]))
        # self.likelihood_space[-1].append([lnl - p, *pars] )
        return lnl - p

    def check_delta_pars(self, current_pars):
        print('Callback...')
        pars = np.asarray(current_pars)
        if self.last_pars is None:
            self.last_pars = pars
        else :
            differences = pars - self.last_pars
            if np.any(differences) :
                print(differences)

    def lnl3(self,pars_,x,ydata,yerr,prior,mle_parameters,free_par_indices,fixed_par_index,fixed_par_trial_value):
        all_pars = self.subpx_pars.copy()
        all_pars[free_par_indices] = pars_
        # print(*zip(free_par_indices, all_pars[free_par_indices], pars_), sep='\n')
        all_pars[fixed_par_index] = fixed_par_trial_value
        # print(all_pars[sorted(free_par_indices + [fixed_par_index])])
        lnl = self.lnl(all_pars,x,ydata,yerr)
        p = np.ma.sum(np.ma.log(prior[1:]/prior[0]))
        # print(lnl)
        return lnl - p

    def profile_likelihood(self, column_index, mle_parameters, min_negloglike, ydata, yerr, target_conf=0.68):
        """
        Estimate parameter uncertainties using the change in the profiled likelihood
        function for that parameter.
        """
        print('Estimating profile likelihood uncertainties ({}% confidence) for column {}...'.format(target_conf*100, column_index))
        self.profile_uncertainties.append([])
        target_delta = 0.5*scipy.stats.chi2.ppf(target_conf, 1)
        all_free_par_indices = range(1, len(mle_parameters), 2)
        for par_index, parameter in zip(all_free_par_indices, mle_parameters[all_free_par_indices]):
            print('Parameter {}. MLE is {}. logL(MLE) is {}...'.format(
                par_index, parameter, min_negloglike))
            free_par_indices = list(
                set(all_free_par_indices) - set([par_index]))
            # self.profile_uncertainties[-1].append([])
            delta_lnl = lambda trial_parameter_value: scipy.optimize.minimize(self.lnl3,
                                                                              x0=mle_parameters[
                                                                                      free_par_indices],
                                                                              bounds=[
                                                                                      (0, 5), ] * len(free_par_indices),
                                                                              args=(self.x,
                                                                                    ydata,
                                                                                    yerr,
                                                                                    self.prior[
                                                                                        :, column_index],
                                                                                    mle_parameters,
                                                                                    free_par_indices,
                                                                                    par_index,
                                                                                    trial_parameter_value))['fun'] - min_negloglike - target_delta

            def find_max_parameter(parameter):
                if parameter <= 0 or not np.isfinite(parameter):
                    parameter = 1e-5
                funcVal = delta_lnl(parameter)
                while funcVal < 0  and parameter < 1e5:
                    parameter *= 10
                    funcVal = delta_lnl(parameter)
                return parameter if funcVal > 0 else None

            max_parameter = find_max_parameter(parameter)
            if max_parameter is not None and np.isfinite(max_parameter) :
                print('Trial bracket: f({}) = {}, f({}) = {}'.format(parameter, delta_lnl(parameter), max_parameter, delta_lnl(max_parameter)))
                bound, details = scipy.optimize.brentq(delta_lnl, parameter, max_parameter, full_output=True)
                # bound = scipy.optimize.newton(delta_lnl, x0=parameter)
                self.profile_uncertainties[-1].append((bound, parameter, details))
            else :
                print('Could not locate bound. Returning NaN.')
                self.profile_uncertainties[-1].append((np.nan, parameter, False))
        return self.profile_uncertainties[-1]

    def map_likelihood(self, column_index, mle_parameters, min_negloglike, ydata, yerr, maxpars, minpars=None, parsteps=50) :
        """
        Perform a grid search around the MLE to attempt to map the likelihood space.
        For each parameter of interest (normalizations of model components) step over a
        grid of values and re-fit with that parameter fixed.
        """
        print('Mapping likelihood space for column {}...'.format(column_index))
        self.likelihood_space.append([])
        all_free_par_indices = range(1, len(mle_parameters), 2)
        if minpars is None :
            minpars = mle_parameters.copy()[all_free_par_indices]

        # print(minpars, maxpars, sep='\n\n')

        for par_index, parameter, minpar, maxpar in zip(all_free_par_indices, mle_parameters[all_free_par_indices], minpars, maxpars) :
            print('Parameter {}. MLE is {}. logL(MLE) is {}...'.format(par_index, parameter, min_negloglike))
            self.likelihood_space[-1].append([])
            # negloglike = min_negloglike
            num_steps = 0
            max_delta_le = 0
            free_par_indices = list(set(all_free_par_indices) - set([par_index]))
            trial_values = np.linspace(start=minpar, stop=maxpar, num=parsteps, endpoint=True)

            for trial_parameter_value in trial_values :
                fit_result = scipy.optimize.minimize(self.lnl3,
                                                     x0=mle_parameters[free_par_indices],
                                                     bounds=[(0, 5),]*len(free_par_indices),
                                                     args=(self.x,
                                                           ydata,
                                                           yerr,
                                                           self.prior[:,column_index],
                                                           mle_parameters,
                                                           free_par_indices,
                                                           par_index,
                                                           trial_parameter_value))
                self.likelihood_space[-1][-1].append((trial_parameter_value,
                                                      fit_result['fun'],
                                                      fit_result['fun'] - min_negloglike,
                                                      fit_result['success']))

    def clean_grism(self, use_profile_errors=False, map_likelihood=False):
        x0 = self.subpx_pars[1::2]
        bounds = [(0,5),] * len(x0)

        for i in range(self.grism_len):
            # operate column-by-column...
            ydata, yerr = self.grism_img_s[:,i], self.grism_err_s[:,i]
            if not all(yerr==0.) and any(np.isfinite(ydata)):
                # When minimizing the negative log likelihood the inverse hessian corresponds to the
                # inverse of the Fisher information matrix which is an estimator of the asymptotic
                # covariance matrix.
                fit_result = scipy.optimize.minimize(self.lnl2,x0=x0,bounds=bounds,args=(self.x,ydata,yerr,self.prior[:,i]))
                pars_, inverse_hessian, max_like = fit_result['x'], fit_result['hess_inv'], fit_result['fun']
                # print(inverse_hessian.todense().shape)
                pars = self.subpx_pars.copy()
                pars[1::2] = pars_
                self.grism_model[:,i] = contam_model(self.x,pars,self.c_prof,self.c_cen)
                self.grism_contam_model_pars[i] = pars[[0,*range(3, len(pars), 2)]]
                self.grism_contam_model_covmats[i] = np.sqrt(inverse_hessian.todense())
                # print('covmat', repr(self.grism_contam_model_covmats[i]))
                # print('norms,errors', *list(zip(self.grism_contam_model_pars[i,:], self.grism_contam_model_covmats[i][np.diag_indices_from(self.grism_contam_model_covmats[i])])), sep='\n')
                model_par_uncertainties = np.diag(self.grism_contam_model_covmats[i])[1:]
                if use_profile_errors is None or (use_profile_errors and i in use_profile_errors):
                    profile_uncertainties = self.profile_likelihood(i, pars.copy(), max_like, ydata, yerr)
                    model_par_uncertainties = np.asarray(profile_uncertainties)[:,0] - self.grism_contam_model_pars[i]
                    model_par_uncertainties = np.where(model_par_uncertainties > 0, model_par_uncertainties, np.asarray(profile_uncertainties)[:,0])
                # self.uncertain_grism_model[:,i] = uncertain_contam_model(self.x,pars,np.sqrt(np.diag(self.grism_contam_model_covmats[i])[1:]),self.c_prof,self.c_cen)
                print('UNCERTAINTIES:', len(model_par_uncertainties), model_par_uncertainties)
                print('PARAMETERS:', len(self.grism_contam_model_pars[i]), self.grism_contam_model_pars[i])

                self.uncertain_grism_model[:,i] = uncertain_contam_model(self.x,pars,model_par_uncertainties,self.c_prof,self.c_cen)
                if map_likelihood is None or (map_likelihood and i in map_likelihood) :
                    self.map_likelihood(i, pars.copy(), max_like, ydata, yerr, maxpars=np.asarray([profileBoundData[0] for profileBoundData in self.profile_uncertainties[-1]]) * 1.5)
        print ("Grism cleaning done for Obj #%i" % self.entry['NUMBER'])
        self.grism_img_c = self.grism_img - self.grism_model
        self.uncertain_grism_img_c = self.uncertain_grism_img - self.uncertain_grism_model

    def lnl_minuit(self,*args):
        pars_,x,ydata,yerr,prior
        # Retrieve best fitting parameters derived from subpixelling - only the subset
        # Corresponding to the normalizations of the target and contamination models
        # (supplied as _pars) should be varied so...
        pars = self.subpx_pars.copy()
        # Update the subpixelling best fit parameters with the current trial normalization
        # values...
        pars[1::2] = pars_
        # Now compute the likelihood of the model given those parameters.
        lnl = self.lnl(pars,x,ydata,yerr)
        # One can estimate the prior expectation of flux level for the column based on the
        # grism sensitivity curve and the expected positions of the 1st order dispersion
        # flux for each contaminant and the target.
        #
        # ** The units of the sensitivity curve are e−/s per erg cm^{−2} s^{−1} Ang.^{−1}
        # ** The prior describes the superposed sensitivity curve values for each pixel in a
        # single image column.
        # ** The zeroth element of the prior original unshifted sensitivity curve.
        p = np.ma.sum(np.ma.log(prior[1:]/prior[0]))
        # self.likelihood_space[-1].append([lnl - p, *pars] )
        return lnl - p

    def clean_grism_minuit(self, use_profile_errors=False, map_likelihood=False):
        x0 = self.subpx_pars[1::2]
        bounds = [(0,5),] * len(x0)

        for i in range(self.grism_len):
            # operate column-by-column...
            ydata, yerr = self.grism_img_s[:,i], self.grism_err_s[:,i]
            if not all(yerr==0.) and any(np.isfinite(ydata)):
                # When minimizing the negative log likelihood the inverse hessian corresponds to the
                # inverse of the Fisher information matrix which is an estimator of the asymptotic
                # covariance matrix.
                fit_result = scipy.optimize.minimize(self.lnl2,x0=x0,bounds=bounds,args=(self.x,ydata,yerr,self.prior[:,i]))
                pars_, inverse_hessian, max_like = fit_result['x'], fit_result['hess_inv'], fit_result['fun']
                # print(inverse_hessian.todense().shape)
                pars = self.subpx_pars.copy()
                pars[1::2] = pars_
                self.grism_model[:,i] = contam_model(self.x,pars,self.c_prof,self.c_cen)
                self.grism_contam_model_pars[i] = pars[[0,*range(3, len(pars), 2)]]
                self.grism_contam_model_covmats[i] = np.sqrt(inverse_hessian.todense())
                # print('covmat', repr(self.grism_contam_model_covmats[i]))
                # print('norms,errors', *list(zip(self.grism_contam_model_pars[i,:], self.grism_contam_model_covmats[i][np.diag_indices_from(self.grism_contam_model_covmats[i])])), sep='\n')
                model_par_uncertainties = np.diag(self.grism_contam_model_covmats[i])[1:]
                if use_profile_errors is None or (use_profile_errors and i in use_profile_errors):
                    profile_uncertainties = self.profile_likelihood(i, pars.copy(), max_like, ydata, yerr)
                    model_par_uncertainties = np.asarray(profile_uncertainties)[:,0] - self.grism_contam_model_pars[i]
                    model_par_uncertainties = np.where(model_par_uncertainties > 0, model_par_uncertainties, np.asarray(profile_uncertainties)[:,0])
                # self.uncertain_grism_model[:,i] = uncertain_contam_model(self.x,pars,np.sqrt(np.diag(self.grism_contam_model_covmats[i])[1:]),self.c_prof,self.c_cen)
                print('UNCERTAINTIES:', len(model_par_uncertainties), model_par_uncertainties)
                print('PARAMETERS:', len(self.grism_contam_model_pars[i]), self.grism_contam_model_pars[i])

                self.uncertain_grism_model[:,i] = uncertain_contam_model(self.x,pars,model_par_uncertainties,self.c_prof,self.c_cen)
                if map_likelihood is None or (map_likelihood and i in map_likelihood) :
                    self.map_likelihood(i, pars.copy(), max_like, ydata, yerr, maxpars=np.asarray([profileBoundData[0] for profileBoundData in self.profile_uncertainties[-1]]) * 1.5)
        print ("Grism cleaning done for Obj #%i" % self.entry['NUMBER'])
        self.grism_img_c = self.grism_img - self.grism_model
        self.uncertain_grism_img_c = self.uncertain_grism_img - self.uncertain_grism_model


    def save_fits(self):
        hdulist = fitsio.HDUList([])
        for data,extname in zip([self.grism_img,self.grism_err,self.grism_model,self.grism_img_c],
                                    ['SCI','ERR','MODEL','CLEAN']):
            data = data.filled(fill_value=np.NaN)
            hdu = fitsio.PrimaryHDU(header=self.grism_hdr,data=data)
            hdu.name = extname
            hdulist.append(hdu)
        hdulist.writeto(self.output_dir+'Par%i/clean/aXeWFC3_%s_clean_ID%s.fits' % (self.par_num,self.grism,self.obj_num),overwrite=True)

    def process(self, use_profile_errors=False, map_likelihood=False):

        self.get_subpx_shifts()
        self.clean_grism(use_profile_errors, map_likelihood)
        self.save_fits()

        plot_subpx_shifts(self.grism,self.entry,self.subpx_pars,self.xf,self.grism1D_img,self.model_args,self.output_dir)
        plot_clean_grism(self.grism,self.entry,self.grism_hdr,self.grism_img,self.grism_model,self.grism_img_c,
                         self.contams,self.subpx_pars[2::2],self.px_scale_cor,self.output_dir)
        plot_prior(self.prior, self.grism_hdr)
