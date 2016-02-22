import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tools import *

def plot_subpx_shifts(grism,entry,pars,xf,yf,model_args,output_dir):

    d_prof,d_cen,c_prof,c_cen = model_args

    fig,ax = plt.subplots(1,1,figsize=(15,6),dpi=75,tight_layout=True)
    ax.axhline(continuum(xf,pars),c='g')
    ax.plot(xf,direct_model(xf,pars,d_prof,d_cen),c='r')
    for p,_ in zip(c_prof,c_cen):
        ax.plot(xf,contam_model(xf,pars,c_prof,c_cen),c='b')
        ax.vlines(_,0.02,0.03,color='b',lw=1.5)
    ax.plot(xf,profile_model(xf,pars,model_args),c='k',ls='--')
    ax.plot(xf,yf,c='k')
    ax.set_xlim(min(xf),max(xf))
    fig.savefig(output_dir+'/plots/profile_%s_%i.png' % (grism,entry['NUMBER']))

def plot_clean_grism(grism,entry,grism_hdr,grism_img,grism_model,grism_img_clean,contams,subpx_shft,px_scale_cor,output_dir):

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,8),sharex=True,sharey=True)
    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.15,top=0.98,hspace=0)
    c,r = np.indices(grism_img.shape)
    r = get_waves(r,grism_hdr)

    sig = np.std(grism_img)
    sig = np.std(np.clip(grism_img,-5*sig,5*sig))

    im = ax1.pcolormesh(r,c,grism_img,vmin=-3*sig,vmax=3*sig,cmap=plt.cm.viridis)
    ax2.pcolormesh(r,c,grism_model,vmin=-3*sig,vmax=3*sig,cmap=plt.cm.viridis)
    ax3.pcolormesh(r,c,grism_img_clean,vmin=-3*sig,vmax=3*sig,cmap=plt.cm.viridis)

    cbaxes = fig.add_axes([0.05, 0.05, 0.9, 0.02])
    cbax   = fig.colorbar(mappable=im, cax=cbaxes, orientation='horizontal')

    if grism_hdr['CRVAL1'] < 1e4:
        xc, dx = 8000., 11750. - 8000.
        ax3.set_xlim(7250,12750)
    else:
        xc, dx = 10500., 17500. - 10500.
        ax3.set_xlim(9500,18500)

    ax3.set_xlabel('Wavelength [$\AA$]')
    ax3.set_ylim(0,grism_img.shape[0])
    _ = [ax.yaxis.set_visible(False) for ax in [ax1,ax2,ax3]]

    yc, dy = grism_img.shape[0]/2., entry['BBOX_Y']*px_scale_cor
    for ax in [ax1,ax2,ax3]:
        ax.add_patch(Rectangle((xc,yc-entry['BBOX_Y']/2.*px_scale_cor+subpx_shft[0]), dx, dy,linewidth=0.5,facecolor='none',edgecolor='w'))
        for contam,subpx in zip(contams,subpx_shft[1:]):
            rx = xc + (contam['X_IMAGE'] - entry['X_IMAGE'])*px_scale_cor * grism_hdr['CDELT1']
            ry = yc + (contam['Y_IMAGE'] - entry['Y_IMAGE'])*px_scale_cor - contam['BBOX_Y']/2.*px_scale_cor+subpx
            ax.add_patch(Rectangle((rx,ry), dx, contam['BBOX_Y']*px_scale_cor,linewidth=0.5,facecolor='none',edgecolor='r'))

    fig.savefig(output_dir+'/plots/grism_%s_%i.png' % (grism,entry['NUMBER']))

def plot_prior(p,hdr):

    fig,ax = plt.subplots(1,1,figsize=(12,6),dpi=75)
    waves = get_waves(np.arange(p.shape[1]), hdr)
    ax.pcolormesh(waves,np.arange(p.shape[0]),p,cmap=plt.cm.viridis)
    ax.set_xlim(min(waves),max(waves))
    ax.set_ylim(p.shape[0],0)
    plt.show()
