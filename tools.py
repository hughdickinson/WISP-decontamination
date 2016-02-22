import numpy as np
import scipy.interpolate
import astropy.io.fits as fitsio
from astropy.nddata import Cutout2D

direct_key = {'G102':'F110','G141':'F160'}

def cut_stamps(img,entry,wcs,hdr,scale):
    pos = (entry['X_IMAGE']-1, entry['Y_IMAGE']-1)
    size = (scale*entry['BBOX_Y'],scale*entry['BBOX_X'])
    stamp = Cutout2D(data=img,position=pos,size=size,wcs=wcs,mode='trim')
    stphdr = stamp.wcs.to_header()
    newhdr = hdr.copy()
    for x in ['CRPIX1','CRPIX2']: newhdr[x] = stphdr[x]
    return stamp.data, newhdr

def inEllipse(x, y, xc, yc, a, b, theta):
    """
    (x,y) is coordinate of point being tested
    the region's (x,y) is center of ellipse
    the region's a and b are semimajor and semiminor axes of ellipse
    the region's theta is rotation angle of ellipse in degrees, ccw from x-axis
    returns Boolean
    """
    thetaRad = theta*np.pi/180
    num1 = np.cos(thetaRad)*(x-xc) + np.sin(thetaRad)*(y-yc)
    num2 = np.sin(thetaRad)*(x-xc) - np.cos(thetaRad)*(y-yc)
    return (np.power(num1,2)/np.power(a,2)) + (np.power(num2,2)/np.power(b,2)) <= 1

def bboxEllipse(a,b,theta):
    """
    the region's (x,y) is center of ellipse
    the region's a and b are semimajor and semiminor axes of ellipse
    the region's theta is rotation angle of ellipse in degrees, ccw from x-axis
    returns x, y, dx, dy
    """
    thetaRad = theta*np.pi/180
    dx = np.sqrt( pow(b*np.sin(thetaRad), 2) + pow(a*np.cos(thetaRad), 2) )
    dy = np.sqrt( pow(a*np.sin(thetaRad), 2) + pow(b*np.cos(thetaRad), 2) )
    return dx, dy

def moving_average(x, s=5) :
    ret = np.ma.cumsum(x, axis=-1)
    ret[:,s:] = ret[:,s:] - ret[:,:-s]
    return ret[:,s - 1:] / s

def get_waves(x,hdr):
    return (x - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1']

def get_direct_contams(catalog,entry,dx,dy):
    xc, yc = entry['X_IMAGE'],entry['Y_IMAGE']
    cond1 = (catalog['X_IMAGE']-xc <=  dx+catalog['BBOX_X'])
    cond2 = (catalog['X_IMAGE']-xc >= -dx-catalog['BBOX_X'])
    cond3 = (catalog['Y_IMAGE']-yc <=  dy+catalog['BBOX_Y'])
    cond4 = (catalog['Y_IMAGE']-yc >= -dy-catalog['BBOX_Y'])
    cond  = cond1 & cond2 & cond3 & cond4
    cond[catalog['NUMBER'] == entry['NUMBER']] = False
    contams = catalog[cond]
    return contams

def get_grism_contams(catalog,entry,dx,dy):
    xc, yc = entry['X_IMAGE'],entry['Y_IMAGE']
    cond1 = (catalog['X_IMAGE']-xc <=  dx)
    cond2 = (catalog['X_IMAGE']-xc >= -dx)
    cond3 = (catalog['Y_IMAGE']-yc <=  dy*.5+catalog['BBOX_Y'])
    cond4 = (catalog['Y_IMAGE']-yc >= -dy*.5-catalog['BBOX_Y'])
    cond  = cond1 & cond2 & cond3 & cond4
    cond[catalog['NUMBER'] == entry['NUMBER']] = False
    contams = catalog[cond]
    return contams

def mask_direct_image(img,hdr,sources,fill_value):
    masked = img.copy()
    for source in sources:
        x, y = source['X_IMAGE'], source['Y_IMAGE']
        dx, dy = source['BBOX_X'], source['BBOX_Y']
        xmin = int(np.floor(max(0, source['X_IMAGE']-source['BBOX_X'])))
        xmax = int(np.ceil(min(img.shape[1], source['X_IMAGE']+source['BBOX_X'])))
        ymin = int(np.floor(max(0, source['Y_IMAGE']-source['BBOX_Y'])))
        ymax = int(np.ceil(min(img.shape[0], source['Y_IMAGE']+source['BBOX_Y'])))
        gx,gy = np.meshgrid(range(xmin,xmax),range(ymin,ymax))
        ix,iy = gx.flatten(), gy.flatten()
        cond = inEllipse(ix,iy,source['X_IMAGE'],source['Y_IMAGE'],2.*source['A_IMAGE'],2.*source['B_IMAGE'],source['THETA_IMAGE'])
        masked[iy[cond],ix[cond]] = fill_value
    return masked

def mk_stamp_profile(entry,px_scale_cor,output_dir,filt):
    stamp = fitsio.getdata('%s/stamps/stamp_%s_%i.fits' % (output_dir,filt,entry['NUMBER']))
    nx = stamp.shape[1]
    stamp_trim = stamp[:,nx/3:2*nx/3]
    stamp_1D = np.sum(stamp_trim,axis=-1) / stamp_trim.shape[1]
    stamp_1D.clip(0)
    pixels = np.arange(len(stamp_1D)) - 0.5*len(stamp_1D)
    pixels = pixels * px_scale_cor
    interp_fn = scipy.interpolate.interp1d(pixels,stamp_1D,bounds_error=False,fill_value=0)
    return interp_fn

def continuum(x,pars):
    contin = pars[0]
    return contin

def direct_model(x,pars,d_prof,d_cen):
    direct = pars[1]*d_prof(x - d_cen + pars[2])
    return direct

def contam_model(x,pars,c_prof,c_cen):
    contam = np.array([A*prof(x - cen + shft) for A,shft,prof,cen in zip(pars[3::2],pars[4::2],c_prof,c_cen)])
    return np.sum(contam,axis=0)

def profile_model(x,pars,model_args):
    d_prof, d_cen, c_prof, c_cen = model_args
    contin = continuum(x,pars)
    direct = direct_model(x,pars,d_prof,d_cen)
    contam = contam_model(x,pars,c_prof,c_cen)
    return contin + direct + contam