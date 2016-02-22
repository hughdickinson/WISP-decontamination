import logging, time, os
import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy import optimize, interpolate, stats

#custom functions
from functions import *

def process(P, F, stamp=False, plot=False):
    '''P is an int refering to the Par being processed
    F is the int 110 or 160; it sets G to 102 or 141, respectively
    if stamp is True, stamps of all the bright objects will be cut and put in a directory called stamps
    if plot is True, a pdf showing the steps of the processing will be created and put in a directory called stages'''
    
    if plot:
        import matplotlib
        matplotlib.use('AGG') #non-interactive backend
        from matplotlib import pyplot as plt
        from matplotlib.colors import SymLogNorm
        logging.info('Plotting enabled')
        
    if F==110:
        G = 102
    elif F==160:
        G = 141
    else:
        raise ValueError('F==%i; should be 110 or 160' % F)
    
    imageDir = 'Par%i/DATA/DIRECT_GRISM' % P
    stampDir = '%s/F%i_stamps' % (imageDir, F)
    grismDir = 'Par%i/G%i_DRIZZLE' % (P, G)
    subgrismDir = '%s/subtracted' % grismDir
    modelDir = '%s/contaminationModels' % subgrismDir
    stageDir = '%s/processingStages' % subgrismDir
    detailsDir = '%s/detailedPlots' % subgrismDir
    if not os.path.exists(stampDir):
        os.mkdir(stampDir)
    if not os.path.exists(subgrismDir):
        os.mkdir(subgrismDir)
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    if not os.path.exists(stageDir):
        os.mkdir(stageDir)
    if not os.path.exists(detailsDir):
        os.mkdir(detailsDir)
    
    #load image data
    image = '%s/F%iW_sci.fits' % (imageDir, F)
    img = fits.getdata(image, 0)
    header = fits.open(image)[0].header
    catalog = getCatalog('%s/fin_F%i.cat' % (imageDir, F))
    distortionFactor = .08/0.128254
    
    #separate list of galaxies by brightness cut-off
    magcutFaint = 24 #these are too faint to bother with
    magcutBright = 20 #point sources (stars) require extra-large stamps
    faint = catalog[catalog['MAG_F1153W']>magcutFaint]
    stars = catalog[(catalog['MAG_F1153W']<=magcutBright) & (catalog['CLASS_STAR']>0.1)]
    bright = catalog[(catalog['MAG_F1153W']<=magcutFaint) & ((catalog['MAG_F1153W']>magcutBright) | (catalog['CLASS_STAR']<=0.1))]
    #fb = np.hstack((bright, faint))
    
    #cut stamps
    if stamp or not os.path.exists('%s/stamp1.fits' % stampDir):
        print 'Cutting stamps'
        for entry in stars:
            cutStamp(img, header, stampDir, entry, catalog, 4)
        for entry in np.hstack((bright, faint)):
            cutStamp(img, header, stampDir, entry, catalog, 2)
    
    for entry in bright[bright['NUMBER']>6]:
        
        #find all entries that might overlap
        startTime = time.time()
        grism = '%s/aXeWFC3_G%i_mef_ID%i.fits' % (grismDir, G, entry['NUMBER'])
        if not os.path.exists(grism):
            print 'Skipping object %i; file not found: %s' % (entry['NUMBER'], grism)
        else:
            print 'grism %i' % entry['NUMBER']
            logging.info('grism %i' % entry['NUMBER'])
            gimg = fits.getdata(grism, 1)
            gerr = fits.getdata(grism, 2)
            gheader = fits.getheader(grism, 1)
            gdx = gheader['NAXIS1']
            gdy = gheader['NAXIS2']
            gyRef = gheader['CRPIX2']
            
            #find contaminants to have own profiles calculated
            cond1 = entry['NUMBER']!=bright['NUMBER']
            cond2 = entry['Y_IMAGE']+gdy/2. >= bright['Y_IMAGE']-bright['A_IMAGE']
            cond3 = entry['Y_IMAGE']-gdy/2. <= bright['Y_IMAGE']+bright['A_IMAGE']
            cond4 = entry['X_IMAGE']-gdx/2. <= bright['X_IMAGE']+gdx/2.
            cond5 = bright['X_IMAGE']+gdx/2. <= entry['X_IMAGE']+gdx/2.
            cond6 = entry['X_IMAGE']-gdx/2. <= bright['X_IMAGE']-gdx/2.
            cond7 = bright['X_IMAGE']-gdx/2. <= entry['X_IMAGE']+gdx/2.
            contams = bright[cond1 & cond2 & cond3 & ((cond4 & cond5) | (cond6 & cond7))]
            print 'contaminants: %s' % contams['NUMBER']
            logging.info('contaminants: %s' % contams['NUMBER'])
            
            #get main profile
            stamp = '%s/stamp%i.fits' % (stampDir, entry['NUMBER'])
            simg = fits.getdata(stamp, 0)
            sheader = fits.getheader(stamp, 0)
            sdx = sheader['NAXIS1']
            sdy = sheader['NAXIS2']
            stripe = [] #linear profile starting from bottom of stamp
            for row in simg:
                val = 0
                for x, col in enumerate(row):
                    if sdx/3. <= x <= sdx*2/3.: #use middle third of profile
                        val += col
                stripe.append(val)
            
            #extend stripe to height of grism
            speakLocationY = (np.where(stripe == max(stripe))[0][0])
            profile = np.zeros(gdy)
            profile[gyRef-speakLocationY:gyRef-speakLocationY+sdy] = stripe
            
            #normalize and interpolate the profile
            profileInterp = interpolate.interp1d(xrange(gdy), profile/gdy, kind='linear', bounds_error=False, fill_value=0)
            pRange = np.array(xrange(100*(gdy-1)+1))/100.
            profile = profileInterp(pRange)
            
            #get contaminating profiles and relative x range
            c_bounds, c_profiles, c_profileInterps = {}, {}, {}
            for contam in contams:
                c_stamp = '%s/stamp%i.fits' % (stampDir, contam['NUMBER'])
                c_simg = fits.getdata(c_stamp, 0)
                c_sheader = fits.getheader(c_stamp, 0)
                c_sdx = c_sheader['NAXIS1']
                c_sdy = c_sheader['NAXIS2']
                c_xoffset = (contam['X_IMAGE'] - entry['X_IMAGE'])*distortionFactor #why distorted?
                c_yoffset = (contam['Y_IMAGE'] - entry['Y_IMAGE'])*distortionFactor #why distorted?
                c_stripe = [] #linear profile starting from bottom of stamp
                for row in c_simg:
                    val = 0
                    for x, col in enumerate(row):
                        if c_sdx/3. <= x <= c_sdx*2/3.: #use middle third of profile
                            val += col
                    c_stripe.append(val)
                
                #make the stripe a spline to allow subpixelling to 0.01 pix
                c_spline = interpolate.interp1d(xrange(c_sdy), c_stripe, kind='linear', bounds_error=False, fill_value=0)
                c_fineRange = np.array(xrange(100*c_sdy))/100.
                c_splineVals = c_spline(c_fineRange)
                
                #extend stripe to height of grism
                c_speakLocationY = (np.where(c_stripe == max(c_stripe))[0][0]) #use data, not interpolation
                c_profile = np.zeros(100*gdy)
                left = int(round((gyRef-c_speakLocationY+c_yoffset)*100))
                right = int(round((gyRef-c_speakLocationY+c_sdy+c_yoffset)*100))
                c_profile[max(left,0):min(right,gdy*100)] = c_splineVals[-min(left,0):min(right,gdy*100)-left]
                
                #interpolate the profile
                c_profileInterp = interpolate.interp1d(np.arange(len(c_profile))/100., c_profile/gdy, kind='linear', bounds_error=False, fill_value=0)
                c_profile = c_profileInterp(pRange)
                
                #determine to where in grism the contamination may extend
                #needs improvement; currently too generous
                cdx = [max(0, c_xoffset-contam['A_IMAGE']), min(c_xoffset+gimg.shape[1]+contam['A_IMAGE'], gdx-1)]
                
                c_bounds[contam['NUMBER']] = cdx
                c_profiles[contam['NUMBER']] = c_profile
                c_profileInterps[contam['NUMBER']] = c_profileInterp
            
            #code to compress whole grism
            gimgMasked = np.ma.masked_array(gimg, mask=(gerr==0), fill_value=np.NaN)
            gerrMasked = np.ma.masked_array(gerr, mask=(gerr==0), fill_value=np.NaN)
            gColumnTotal = np.ma.average(gimgMasked, axis=-1)
            gErrColumnTotal = np.sqrt(np.ma.average(np.square(gerrMasked), axis=-1))
            gColumnTotalFine = interpolate.interp1d(xrange(gdy), gColumnTotal, kind='linear', bounds_error=False, fill_value=0)(pRange)
            gErrColumnTotalFine = interpolate.interp1d(xrange(gdy), gErrColumnTotal, kind='linear', bounds_error=False, fill_value=0)(pRange)
            gColumnTotalFine = np.ma.masked_array(gColumnTotalFine, mask=(gErrColumnTotalFine==0), fill_value=np.NaN)
            gErrColumnTotalFine = np.ma.masked_array(gErrColumnTotalFine, mask=(gErrColumnTotalFine==0), fill_value=np.NaN)
            
            #list all the profiles
            profiles = [profileInterp]
            for c in c_profileInterps:
                profiles.append(c_profileInterps[c])
            
            #find the pixel offsets for the whole object
            weights = [1,0] * len(profiles) #both amplitudes and dy pixel offsets can vary
            b = [(0, 10), (-2, 2)] * len(profiles) #bound amplitudes to be nonnegative, pixel offset within 2
            st = time.time()
            minimization = optimize.minimize(residualVaryOffset, weights, args=(gColumnTotalFine, gErrColumnTotalFine, profiles, gdy), bounds=b)
            et = time.time()
            logging.info(minimization)
            logging.info('Time to optimize: %f' % (et-st))
            weights = minimization['x']
            offsets = weights[1::2]
            success = minimization['success']
            if not success:
                print 'Overall minimization unsuccessful'
            
            #plot the data and models
            if plot:
                plotDir = '%s/%i' % (detailsDir, entry['NUMBER'])
                if not os.path.exists(plotDir):
                    os.mkdir(plotDir)
                
                f, (dataPlot, residPlot) = plt.subplots(2)
                dataPlot.plot(gimg, alpha=0.1)
                dataPlot.plot(pRange, gColumnTotalFine, label='mean')
                weightedProfArrays = makeModels(profiles, weights[0::2], offsets, gdy)
                outputModel = np.zeros(len(pRange))
                for profNum, prof in enumerate(weightedProfArrays):
                    dataPlot.plot(pRange, prof, label='model %i' % profNum)
                    outputModel += prof
                
                dataPlot.legend(loc='best')
                dataPlot.set_ylabel('intensity')
                dataPlot.set_title('Compressed grism: %s\nweights = %s' % ('success' if success else 'failure', weights))
                residPlot.plot(pRange, gColumnTotalFine - outputModel, label='residual')
                residPlot.legend(loc='best')
                residPlot.set_xlabel('y (pixels)')
                residPlot.set_ylabel('residual')
                plt.savefig('%s/compressedGrism.png' % plotDir)
                plt.close(f)
            
            #determine contributions due to each object via chi^2 minimization
            #each column point is the mean of the three points centered around its index
            contamGimg = np.ma.masked_array(np.zeros(gimg.shape), mask=np.ma.getmask(gimgMasked))
            subtractGimg = np.ma.copy(gimgMasked)
            for x in xrange(gdx):
                if x==0: #nothing left of first index
                    temp = np.ma.masked_array([gimgMasked.T[x], gimgMasked.T[x+1]])
                    tempErr = np.ma.masked_array([gerrMasked.T[x], gerrMasked.T[x+1]])
                elif x==gdx-1: #nothing right of last index
                    temp = np.ma.masked_array([gimgMasked.T[x-1], gimgMasked.T[x]])
                    tempErr = np.ma.masked_array([gerrMasked.T[x-1], gerrMasked.T[x]])
                else:
                    temp = np.ma.masked_array([gimgMasked.T[x-1], gimgMasked.T[x], gimgMasked.T[x+1]])
                    tempErr = np.ma.masked_array([gerrMasked.T[x-1], gerrMasked.T[x], gerrMasked.T[x+1]])
                
                gColumn = np.ma.average(temp.T, axis=-1)
                gErrColumn = np.sqrt(np.ma.average(tempErr.T, axis=-1))
                gColumnFine = interpolate.interp1d(xrange(gdy), gColumn, kind='linear', bounds_error=False, fill_value=0)(pRange)
                gErrColumnFine = interpolate.interp1d(xrange(gdy), gErrColumn, kind='linear', bounds_error=False, fill_value=0)(pRange)
                gColumnFine = np.ma.masked_array(gColumnFine, mask=(gErrColumnFine==0), fill_value=np.NaN)
                gErrColumnFine = np.ma.masked_array(gErrColumnFine, mask=(gErrColumnFine==0), fill_value=np.NaN)
                
                #determine which contaminants may be contaminating the grism
                profiles = [profileInterp]
                for c in contams['NUMBER']:
                    if c_bounds[c][0] <= x <= c_bounds[c][1]:
                        profiles.append(c_profileInterps[c])
                
                #determine the weights of each profile and calculate their contributions
                weights = [1] * len(profiles) #both amplitudes and dy pixel offsets can vary
                b = [(0, 10)] * len(profiles) #bound amplitudes to be nonnegative, pixel offset within 2
                minimization = optimize.minimize(residualConstOffset, weights, args=(offsets, gColumnFine, gErrColumnFine, profiles, gdy), bounds=b)
                if not minimization['success']:
                    print 'Minimization unsuccessful at x=%i' % x
                    logging.info('Minimization unsuccessful at x=%i' % x)
                    #raise Warning(minimization)
                
                weights = minimization['x']
                weightedProfArrays = makeModels(profiles, weights, offsets, gdy)
                logging.info('At x = %i: weights = %s, success = %s' % (x, weights, minimization['success']))
                
                #plot the data and models
                if plot and not x%(gdx/10): #only plot 10 x values
                    f, (dataPlot, residPlot) = plt.subplots(2)
                    dataPlot.plot(pRange, gColumnFine, label='observed')
                    model = np.zeros(len(pRange))
                    for profNum, prof in enumerate(weightedProfArrays):
                        dataPlot.plot(pRange, prof, label='model %i' % profNum)
                        model += prof
                    
                    dataPlot.legend(loc='best')
                    dataPlot.set_ylabel('intensity')
                    dataPlot.set_title('At x = %i: %s\nweights = %s' % (x, 'success' if minimization['success'] else 'failure', weights))
                    residPlot.plot(pRange, gColumnFine - model, label='residual')
                    residPlot.legend(loc='best')
                    residPlot.set_xlabel('y (pixels)')
                    residPlot.set_ylabel('residual')
                    plt.savefig('%s/grismAt%i.png' % (plotDir, x))
                    plt.close(f)
                
                #subtract each contaminating profile from the grism
                for y, ySlice in enumerate(gimg):
                    for profNum, prof in enumerate(weightedProfArrays):
                        if profNum and gimg[y,x]:
                            subtractGimg[y,x] -= prof[::100][y]
                
                #creates a contamination model
                for y, val in enumerate(gimg):
                    for profNum, prof in enumerate(weightedProfArrays):
                        if gimg[y,x]:
                            contamGimg[y,x] += prof[::100][y]
            
            #save the new grism
            subtractFile = '%s/aXeWFC3_G%i_mef_ID%i_subtracted.fits' % (subgrismDir, G, entry['NUMBER'])
            fits.writeto(subtractFile, data=subtractGimg.data, header=gheader, clobber=True)
            
            if plot:
                contamFile = '%s/contam%i.fits' % (modelDir, entry['NUMBER'])
                fits.writeto(contamFile, data=contamGimg.data, header=gheader, clobber=True)
                
                xmin, xmax, ymin, ymax = max(entry['X_IMAGE']-gdx/2., 0), min(entry['X_IMAGE']+gdx/2., img.shape[1]-1), \
                                         max(entry['Y_IMAGE']-gdy/2., 0), min(entry['Y_IMAGE']+gdy/2., img.shape[0]-1)
                
                f, ((directImagePlot, directImageProfilePlot), (grismPlot, grismProfilePlot), (contamPlot, contamProfilePlot), \
                    (subtractedPlot, subtractedProfilePlot)) = plt.subplots(4, 2, sharey=True)

                #plot a grism-sized piece of the direct image
                crop = img[ymin:ymax, xmin:xmax]
                directImagePlot.imshow(crop, norm=SymLogNorm(0.1))
                if success:
                    directImagePlot.set_title('object %i' % entry['NUMBER'])
                else:
                    directImagePlot.set_title('object %i (minimization unsuccessful)' % entry['NUMBER'])
                
                directImagePlot.set_ylabel('direct')
                directImagePlot.axis([0, gdx, 0, gdy])
                directImageProfile = profile
                for profNum in c_profiles:
                    directImageProfile += c_profiles[profNum]
                
                directImageProfilePlot.plot(directImageProfile, pRange)
                directImageProfilePlot.set_title('profiles')
                
                #plot the measured grism and its intensity profile
                vmin = np.ma.min(gimgMasked)
                vmax = np.ma.max(gimgMasked)
                grismPlot.imshow(gimgMasked, norm=SymLogNorm(0.1, vmin=vmin, vmax=vmax))
                grismPlot.set_ylabel('grism')
                grismProfilePlot.plot(gColumnTotalFine, pRange)
                
                #plot the contamination model and its intensity profile
                contamPlot.imshow(contamGimg, norm=SymLogNorm(0.01))
                contamPlot.set_ylabel('model')
                contamProfilePlot.plot(outputModel, pRange)
                
                #plot the subtracted grism and its intensity profile
                subtractedPlot.imshow(subtractGimg, norm=SymLogNorm(0.1, vmin=vmin, vmax=vmax))
                subtractedPlot.set_ylabel('subtracted')
                subtractedProfile = np.ma.average(subtractGimg, axis=-1)
                subtractedProfilePlot.plot(subtractedProfile, xrange(gdy))
                
                plt.savefig('%s/stages%i.png' % (stageDir, entry['NUMBER']))
                plt.close(f)
        
        endTime = time.time()
        logging.info('Total time for object %i: %f' % (entry['NUMBER'], endTime-startTime))

if __name__ == '__main__':
    #when run from command line, run full process on selected Pars
    import argparse, ast
    
    logging.basicConfig(filename='grismProcessing.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.captureWarnings(True)
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--stamp', help='create new stamps of all objects; always true on initial run', action='store_true')
        parser.add_argument('-p', '--plot', help='create PDFs with processing stages, i.e. before/after grisms, contamination models, intensity profiles', \
                            action='store_true')
        parser.add_argument('--par', help='argument is either individual Par or list of them to process; if none provided, all Pars will be processed')
        args = parser.parse_args()
        
        allPars = []
        for d in next(os.walk('.'))[1]:
            if d[0:3] == 'Par':
                allPars.append(int(d[3:]))
        
        if not args.par:
            if raw_input('Process all %i Pars? (y/n) ' % len(allPars)).lower() == 'y':
                ParList = allPars
            else:
                exit()
        else:
            pars = ast.literal_eval(args.par)
            if type(pars) is int:
                if pars in allPars:
                    ParList = [pars]
                else:
                    raise ValueError('Par%i does not exist' % pars)
            elif type(pars) is list:
                for p in pars:
                    if p not in allPars:
                        raise ValueError('Par%i does not exist' % p)
                ParList = pars
            else:
                raise ValueError('Argument is %s, not int or list' % type(pars))

        start = time.time()
        for P in ParList:
            for F in [110, 160]:
                print 'Processing Par%i, F%i' % (P, F)
                process(P, F, stamp=args.stamp, plot=args.plot)
        end = time.time()
        logging.info('Total time to process %i pars: %f' % (len(ParList), (end-start)))
    
    except BaseException as e:
        logging.exception(e)
        raise e
