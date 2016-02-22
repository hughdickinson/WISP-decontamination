'''contains custom functions for use with grismProcessing'''

import numpy as np
import os, csv
from astropy import wcs
from astropy.io import fits
from astropy.nddata import Cutout2D

def getCatalog(catFile):
    '''converts the associated .cat file into a dict for easy look-up
    doubles the ellipse axes because the values in the catalog are half the image size
    adds bounding box values for use in other functions'''
    
    catalogHeader = [('RA_DEC_NAME','>S25'), ('NUMBER',int), ('X_IMAGE',float), ('Y_IMAGE',float),('A_IMAGE',float), ('B_IMAGE',float), ('THETA_IMAGE',float), \
                     ('X_WORLD',float), ('Y_WORLD',float), ('A_WORLD',float), ('B_WORLD',float), ('THETA_WORLD',float), ('MAG_F1153W',float), ('MAGERR_AUTO',float), \
                     ('CLASS_STAR',float), ('FLAGS',int)]
    catalogRaw = np.genfromtxt(catFile, dtype=catalogHeader,)
    doubleThese = ['A_IMAGE', 'B_IMAGE', 'A_WORLD', 'B_WORLD']
    for this in doubleThese:
        catalogRaw[this] *= 2
    dtyp = catalogRaw.dtype.descr + [('BBOX_X',float), ('BBOX_Y',float)]
    catalog = np.recarray(len(catalogRaw), dtype=dtyp)
    for x in catalogRaw.dtype.names:
        catalog[x] = catalogRaw[x]
    bbox_x, bbox_y = bboxEllipse(catalogRaw['A_IMAGE'], catalogRaw['B_IMAGE'], catalogRaw['THETA_IMAGE'])
    catalog['BBOX_X'] = bbox_x
    catalog['BBOX_Y'] = bbox_y
    
    return catalog

def inEllipse(x, y, xc, yc, a, b, theta):
    '''(x,y) is the coordinate being tested
    (xc,yc) is the center of the ellipse
    a and b are semimajor and semiminor axes of the ellipse
    theta is the rotation angle of the ellipse in degrees, ccw from x-axis
    returns Boolean'''
    thetaRad = theta*np.pi/180
    num1 = np.cos(thetaRad)*(x-xc) + np.sin(thetaRad)*(y-yc)
    num2 = np.sin(thetaRad)*(x-xc) - np.cos(thetaRad)*(y-yc)
    return (np.square(num1/a) + np.square(num2/b)) <= 1

def bboxEllipse(a, b, theta):
    '''a and b are the semimajor and semiminor axes of the ellipse
    theta is the rotation angle of the ellipse in degrees, ccw from x-axis
    returns dx, dy'''
    thetaRad = theta*np.pi/180
    dx = np.sqrt( np.square(b*np.sin(thetaRad)) + np.square(a*np.cos(thetaRad)) )
    dy = np.sqrt( np.square(a*np.sin(thetaRad)) + np.square(b*np.cos(thetaRad)) )
    return dx, dy

def cutStamp(img, imgHeader, outdir, entry, catalog, scale=1):
    '''cuts stamps of size scale times the size of the entry and saves them as fits files to outdir
    img and imgHeader are the data array and header, respectively, of the source file'''
    
    xc, yc, dx, dy = entry['X_IMAGE'], entry['Y_IMAGE'], entry['BBOX_X'], entry['BBOX_Y']
    cond1 = catalog['NUMBER'] != entry['NUMBER']
    cond2 = catalog['X_IMAGE']-catalog['BBOX_X'] <= xc+scale*dx
    cond3 = catalog['X_IMAGE']+catalog['BBOX_X'] >= xc-scale*dx
    cond4 = catalog['Y_IMAGE']-catalog['BBOX_Y'] <= yc+scale*dy
    cond5 = catalog['Y_IMAGE']+catalog['BBOX_Y'] >= yc-scale*dy
    cond = cond1 & cond2 & cond3 & cond4 & cond5
    contams = catalog[cond]
    
    img = np.copy(img)
    for contam in contams:
        xmin = int(np.floor(max(0, contam['X_IMAGE']-contam['BBOX_X'])))
        xmax = int(np.ceil(min(img.shape[1], contam['X_IMAGE']+contam['BBOX_X'])))
        ymin = int(np.floor(max(0, contam['Y_IMAGE']-contam['BBOX_Y'])))
        ymax = int(np.ceil(min(img.shape[0], contam['Y_IMAGE']+contam['BBOX_Y'])))
        gx, gy = np.meshgrid(range(xmin,xmax), range(ymin,ymax))
        ix, iy = gx.flatten(), gy.flatten()
        cond = inEllipse(ix, iy, contam['X_IMAGE'], contam['Y_IMAGE'], contam['A_IMAGE'], contam['B_IMAGE'], contam['THETA_IMAGE'])
        img[iy[cond], ix[cond]] = 0#self.sky_median
    
    stamp = Cutout2D(data=img, position=(xc-1,yc-1), size=(scale*dx,scale*dy), wcs=wcs.WCS(imgHeader), mode='trim')
    stampHeader = stamp.wcs.to_header()
    for x in ['CRPIX1', 'CRPIX2']:
        stampHeader[x] = imgHeader[x]
    
    fits.writeto('%s/stamp%i.fits' % (outdir, entry['NUMBER']), data=stamp.data, header=stampHeader, clobber=True)
    
    return None

def residualVaryOffset(weights, observed, error, profiles, dy):
    '''calculate the residual between the observed data and all the profiles
    weights is list of amplitudes and center offsets
    observed and error are ndarrays with measured grism data interpolated to 0.01 pix
    profiles is list of interp1d objects from direct image stamps
    dy is the number of pixels in the observed data
    for use with scipy.optimize.minimize'''
    
    if len(weights) <> 2*len(profiles):
        raise ValueError('wrong number of weights')
    
    #calculate and combine weighted models
    modelList = makeModels(profiles, weights[0::2], weights[1::2], dy)
    model = sum(modelList)
    
    #calculate chi^squared
    r = np.square((observed - model) / error)
    chi2 = np.ma.sum(r)/np.ma.count(r)
    
    #print chi2, weights
    return chi2

def residualConstOffset(weights, offsets, observed, error, profiles, dy):
    '''calculate the residual between the observed data and all the profiles
    weights is list of amplitudes only
    offsets is list of pre-determined center offsets (from functions.residualVaryOffset)
    observed and error are ndarrays with measured grism data interpolated to 0.01 pix
    profiles is list of interp1d objects from direct image stamps
    dy is the number of pixels in the observed data
    for use with scipy.optimize.minimize'''
    
    if len(weights) <> len(profiles):
        raise ValueError('wrong number of weights')
    
    #make and combine weighted models
    modelList = makeModels(profiles, weights, offsets, dy)
    model = sum(modelList)
    
    #calculate chi^squared
    r = np.square((observed - model) / error)
    chi2 = np.ma.sum(r)/np.ma.count(r)
    
    #print chi2, weights
    return chi2

def makeModels(profiles, weights, offsets, dy):
    '''for each interp1d profile, weight, and offset, calculates a scaled model of the intensity curve
    dy is the number of pixels in the observed data
    returns list of evaluated models'''
    
    weightedProfArrays = []
    fineLength = 100*(dy-1)+1
    for profNum, prof in enumerate(profiles):
        model = np.zeros(fineLength)
        offset = offsets[profNum]*100
        lowerBound = int( np.ceil(abs(offset)) if offset<0 else 0 )
        upperBound = int( fineLength - (np.ceil(offset) if offset>0 else 0) )
        pixelRange = (np.arange(lowerBound, upperBound) + offset)/100.
        evaluatedProf = prof(pixelRange)
        model[lowerBound:upperBound] = weights[profNum] * evaluatedProf
        weightedProfArrays.append(model)
    
    return weightedProfArrays
