# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================================
import rasterio
from glob import glob
import numpy as np
import os, time, sys, getopt
from scipy.signal import convolve2d
from tkinter import filedialog
from tkinter import *
# from skimage.io import imsave
from tqdm import tqdm
from tempfile import TemporaryFile
from datetime import datetime
from skimage.filters import threshold_otsu
from skimage.filters.rank import median
from skimage.morphology import disk

# =========================================================
USE_GPU = True
CALC_CONF = True
# =======================================================

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf #numerical operations on gpu
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# =========================================================
# =========================================================
# def rescale(dat,mn,mx):
#    """
#    rescales an input dat between mn and mx
#    """
#    m = np.min(dat.flatten())
#    M = np.max(dat.flatten())
#    return (mx-mn)*(dat-m)/(M-m)+mn


#-----------------------------------
def std_convoluted(image, N):
    """
    fast windowed stdev based on kernel convolution
    """
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    return np.sqrt((s2 - s**2 / ns) / ns)

#-----------------------------------
def z_exp(z,n=0.5):
    """
    empirical exponential scaler
    """
    zmin = np.min(z)
    zmax = np.max(z)
    zrange = zmax-zmin
    ivals = 255*((z-zmin)/(zrange))**n
    return ivals

##==========================================================
def load_ODM_json(weights_file):
    """
    load keras Res-UNet model implementation from json file
    mode=1, dem only
    """
    weights_path = weights_file
    json_file = open(weights_path.replace('.h5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(weights_path)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')#, metrics = [mean_iou, dice_coef])
    return model

#-----------------------------------
def medianfiltmask(est_label, radius=7):
    """
    mask a 1 or 3 band image, band by band for memory saving
    """
    out_label = np.ones_like(est_label)
    for k in range(est_label.shape[-1]):
        out_label[:,:,k] = median(est_label[:,:,k],disk(radius))
    return out_label


#-----------------------------------
def mask(bigtiff,profile,out_mask):
    """
    mask a 1 or 3 band image, band by band for memory saving
    """
    if profile['count']==4:
        bigtiff1 = bigtiff[0,:,:]
        bigtiff1[out_mask==1] = profile['nodata']
        bigtiff[0,:,:] = bigtiff1
        del bigtiff1

        bigtiff2 = bigtiff[1,:,:]
        bigtiff2[out_mask==1] = profile['nodata']
        bigtiff[1,:,:] = bigtiff2
        del bigtiff2

        bigtiff3 = bigtiff[2,:,:]
        bigtiff3[out_mask==1] = profile['nodata']
        bigtiff[2,:,:] = bigtiff3
        del bigtiff3

        bigtiff4 = bigtiff[3,:,:]
        bigtiff4[out_mask==1] = profile['nodata']
        bigtiff[3,:,:] = bigtiff4
        del bigtiff4
    else:
        bigtiff1 = bigtiff[0,:,:]
        bigtiff1[out_mask==1] = profile['nodata']
        bigtiff[0,:,:] = bigtiff1
        del bigtiff1

    return bigtiff

#-----------------------------------
def main(dem, rgb, model_mode, overlap, medfilt_radius, use_otsu): 
    """
    main program
	-o is percentage overlap between tiles (0 for no overlap, 25 for 25% overlap, etc)
	-f is an integer >0 for median filtering. Refers to radius of smoothing kernel. Default = no median filter (0)
	-t is 0 for "no, dont use Otsu - use a threshold of 0.5' and 1 for "yes please, use Otsu"
	 
	Defaults:
	-m  = 1 (dem only - the other model modes are not implemented)
	-o = 25 (percentage overlap)
	-f = 0 (median filter radius)
	-t = 0 (dont use otsu, use threshold=0.5)
    """
    print('.....................................')
    print('.....................................')
    print(datetime.now())
    print("Working on %s" % (dem))
    tilesize = 1024 #
    verbose = True
    # thres = .25  #channel 2 (bad) threshold - deliberately <=.5

    if model_mode==1: #dem only
        print('DEM-only mode')
        weights_file = 'model'+os.sep+'orthoclip_demonly_3class_batch_6.h5'
    elif model_mode==2: #stdev only
        print('STDEV-only mode')
        weights_file = 'model'+os.sep+'orthoclip_stdevonly_3class_batch_6.h5'
    elif model_mode==3: #rgb only
        print('Ortho-only mode')
        weights_file = 'model'+os.sep+'orthoclip_orthoonly_3class_batch_6.h5'
    elif model_mode==4: #rgb+dem
        print('Ortho+DEM mode')
        weights_file = 'model'+os.sep+'orthoclip_RGBdem_3class_batch_6.h5'
    else: #rgb+stdev
        print('Ortho+STDEV mode')
        weights_file = 'model'+os.sep+'orthoclip_RGBstdev_3class_batch_6.h5'


    try:
        os.remove(dem.replace('.tif','_automask.tif'))
    except:
        pass
		
    try:
        os.remove(dem.replace('.tif','_automasked_dem.tif'))
    except:
        pass
			
    do_plot=False #True
    if do_plot:
        import matplotlib.pyplot as plt

    if verbose:
        # # start timer
        start = time.time()

    #====================================================
    with rasterio.open(dem) as src:
        profile = src.profile

    width = profile['width']
    height = profile['height']

    padwidth = width + (tilesize - width % tilesize)
    padheight = height + (tilesize - height % tilesize)

    try:
        #=======================================================
        # step 2: make an instance of the model and load the weights
        model = load_ODM_json(weights_file)
    except:
        print("Model loading failed - check inputs. Exiting ...")
        sys.exit(2)

    #=======================================================
    ##step 3: pre-allocate arrays to be written as big geotiff later, and start filling it
    # by applying the model to windows of data
    print('.....................................')
    print('Using model for prediction on images ...')
    print("Started at %s" % datetime.now())

    try:
        #pre-allocate large arrays to be eventually written to geotiff format files,
        out_mask = np.zeros((padwidth, padheight), dtype=np.uint8)
        accumulator = np.zeros((padwidth, padheight), dtype=np.uint8)
		
        # if CALC_CONF:
            # out_conf = np.ones((padwidth, padheight), dtype=np.float32)

        # counter=0
        # prc_overlap = 25 #100
        prc_overlap = overlap/100
        increment = int((1-prc_overlap)*tilesize)
        # if prc_overlap<.25:
            # increment = tilesize
        # print("Using overlap of %i pixels" % (increment))			
		
        tilesize=1024
        # window1d = np.abs(np.hanning(tilesize/2))
        # window2d = np.sqrt(np.outer(window1d,window1d))
		
        # window2d_up = np.roll(window2d,int(tilesize/4),axis=1)
        # window2d_up[:,:int(tilesize/4)] = 0
		
        for i in tqdm(range(0, padwidth, increment)):
            for j in range(0, padheight, increment):
                window = rasterio.windows.Window(i,j,tilesize, tilesize)

                with rasterio.open(dem) as src:
                    profile = src.profile
                    orig = src.read(window=window).squeeze() #/255.

                if len(np.unique(orig))>1:
                    #print(i,j)
					
                    orig[orig==profile['nodata']] = 0

                    # if do_plot:
                    #     plt.subplot(233); plt.imshow(image, cmap='gray'); plt.axis('off'); plt.colorbar(shrink=.5)
                    if model_mode<3:
                        image = np.squeeze(orig).T
                    elif model_mode==3: #rgb only
                        image = np.squeeze(orig[:3,:,:]).T

                    if model_mode==2: #stdev only
                        ##compute stdev raster using convoluttion with a 3x3 kernel
                        image = std_convoluted(orig, 3)
                        del orig
                        ##scale image exponentially using n=.5
                        image = z_exp(image,n=0.5)

                    if model_mode==4: #rgb + dem
                        image = np.squeeze(orig).T

                        if rgb is not None: #when adding rgb to dem or stdev
                            with rasterio.open(rgb) as src:
                                profile = src.profile
                                orig = src.read(window=window).squeeze() #/255.
                            orig = np.squeeze(orig).T

                    if do_plot:
                        plt.subplot(221); plt.imshow(image, cmap='gray'); plt.axis('off'); plt.colorbar(shrink=.5)

                    do_pad = 0
                    if image.shape[0] < tilesize:

                        if model_mode<3:
                            subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
                            x,y=image.shape
                            subset_pad[:x,:y] = image
                        elif model_mode==3:
                            x,y,z=image.shape
                            subset_pad = np.zeros((tilesize, tilesize,z), dtype=profile['dtype'])
                            subset_pad[:x,:y,:z] = image
                        elif model_mode>3:
                            x,y=image.shape
                            subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
                            subset_pad[:x,:y] = image

                            x,y,z=orig.shape
                            subset_pad2 = np.zeros((tilesize, tilesize,z), dtype=profile['dtype'])
                            subset_pad2[:x,:y,:z] = orig

                        del image
                        image = subset_pad.copy()
                        del subset_pad
                        do_pad = 1						

                        if model_mode>3:
                            if 'subset_pad2' in locals():
                                #del orig
                                orig = subset_pad2.copy()#/255
                                del subset_pad2

                    elif image.shape[1] < tilesize:
                        if model_mode<3:
                            subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
                            x,y=image.shape
                            subset_pad[:x,:y] = image
                        elif model_mode==3:
                            x,y,z=image.shape
                            subset_pad = np.zeros((tilesize, tilesize,z), dtype=profile['dtype'])
                            subset_pad[:x,:y,:z] = image
                        elif model_mode>3:
                            x,y=image.shape
                            subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
                            subset_pad[:x,:y] = image

                            x,y,z=orig.shape
                            subset_pad2 = np.zeros((tilesize, tilesize,z), dtype=profile['dtype'])
                            subset_pad2[:x,:y,:z] = orig
                        do_pad = 1						

                        del image
                        image = subset_pad.copy()#/255
                        del subset_pad

                        if model_mode>3:
                            if 'subset_pad2' in locals():
                                #del orig
                                orig = subset_pad2.copy()#/255
                                del subset_pad2

                    if model_mode>3:
                        image = np.dstack((orig[:,:,0], orig[:,:,1], orig[:,:,2], image))
                        #del orig

                    image = tf.expand_dims(image, 0)
                    #print(image.shape)
                    # use model in prediction mode
                    est_label = model.predict(image , batch_size=1).squeeze()
                    image = image.numpy().squeeze()


                    conf = np.std(est_label, -1)

                    # est_label = np.argmax(est_label, -1) #2=bad, 1=good, 0=bad
                    # est_label = (est_label==1).astype('uint8') #only good

                    est_label_orig = est_label.copy()
                    est_label = np.zeros((est_label.shape[0], est_label.shape[1])) #np.argmax(est_label, -1).astype(np.uint8).T

                    if use_otsu: 
                       otsu_nodata = threshold_otsu(est_label_orig[:,:,0])
                       otsu_baddata = threshold_otsu(est_label_orig[:,:,1])
                       otsu_gooddata = threshold_otsu(est_label_orig[:,:,2])
					   
                       est_label[est_label_orig[:,:,0]>otsu_nodata] = 0  #nodata
                       est_label[est_label_orig[:,:,1]>otsu_baddata] = 0  #bad

                       est_label[est_label_orig[:,:,2]>otsu_gooddata] = 1  #good
                       est_label[est_label>1] = 1
					
                    else:
                       # if np.any(est_label_orig[:,:,2]>.05):
                       est_label[est_label_orig[:,:,0]>0.5] = 0  #nodata
                       est_label[est_label_orig[:,:,1]>0.5] = 0  #bad

                       est_label[est_label_orig[:,:,2]>0.5] = 1  #good
                       est_label[est_label>1] = 1

                    if do_plot:
                        plt.subplot(222); plt.imshow(est_label, cmap='bwr'); plt.axis('off'); plt.colorbar(shrink=.5)
                        plt.subplot(223); plt.imshow(image, cmap='gray'); plt.imshow(est_label, cmap='bwr', alpha=0.3); plt.axis('off'); plt.colorbar(shrink=.5)

                    if do_plot:
                        plt.subplot(224); plt.imshow(conf, cmap='Blues'); plt.axis('off'); plt.colorbar(shrink=.5)
                        plt.savefig('aom_ex'+str(i)+'_'+str(j)+'.png', dpi=300); plt.close()

                    #fill out big rasters
                    if do_pad==1:
                        tilesize_j, tilesize_i = orig.shape
                        out_mask[i:i+tilesize_i,j:j+tilesize_j] += est_label[:tilesize_i,:tilesize_j].astype('uint8') #fill in that portion of big mask					
                    else:
                        out_mask[i:i+tilesize,j:j+tilesize] += est_label.astype('uint8') #fill in that portion of big mask
					
                    #fill out big rasters
                    if do_pad==1:
                        tilesize_j, tilesize_i = orig.shape
                        accumulator[i:i+tilesize_i,j:j+tilesize_j] += np.ones((tilesize_i,tilesize_j)).astype('uint8') #fill in that portion of accumulator
                    else:
                        accumulator[i:i+tilesize,j:j+tilesize] += np.ones((tilesize,tilesize)).astype('uint8') #fill in that portion of accumulator
										

                    del est_label, orig


        #=======================================================
        ##step 4: crop, rotate arrays, divide by N, and write to memory mapped temporary file
        #(i know, classy)
				
        out_mask = out_mask[:width,:height]
        accumulator = accumulator[:width,:height]
		
        accumulator[accumulator<1] = 1		
        out_mask2 = out_mask.copy()		
        out_mask2 = out_mask2/accumulator
        out_mask2[np.isnan(out_mask2)] = 0	
        out_mask2[np.isinf(out_mask2)] = 0	
        out_mask2[out_mask2>0] = 1
        out_mask2 = out_mask2.astype('uint8')
        out_mask = out_mask2.copy()
		
        out_mask = np.rot90(np.fliplr(out_mask))
        out_shape = out_mask.shape

        # out_conf = out_conf[:width,:height]
        # out_conf = np.rot90(np.fliplr(out_conf))

        out_mask = out_mask.astype('uint8')
        out_mask = (out_mask!=1).astype('uint8')
		
        if medfilt_radius>0:
           out_mask = median(out_mask,disk(medfilt_radius))

        # if CALC_CONF:
            # # out_conf = np.divide(out_conf, np.maximum(0,n.astype('float')), out=out_conf, casting='unsafe')
            # out_conf[out_conf==np.nan]=0
            # out_conf[out_conf==np.inf]=0
            # out_conf[out_conf<=0] = 0
            # out_conf = out_conf-1
            # #out_conf[out_conf>.5] = 0
            # # out_conf[out_conf==-1] = 0
            # #out_mask = np.memmap(outfile, dtype='uint8', mode='r', shape=out_shape)
            # out_conf[out_mask==0] = 0

        # # out_mask[out_conf<.25] = 0

        outfile = TemporaryFile()
        fp = np.memmap(outfile, dtype='uint8', mode='w+', shape=out_mask.shape)
        fp[:] = out_mask[:]
        fp.flush()
        del out_mask
        del fp

        # #write out to temporary memory mapped file
        # outfile2 = TemporaryFile()
        # fp = np.memmap(outfile2, dtype='float32', mode='w+', shape=out_conf.shape)
        # fp[:] = out_conf[:]
        # fp.flush()
        # del out_conf
        # del fp

        # #read back in again without using any memory
        # out_conf = np.memmap(outfile2, dtype='float32', mode='r', shape=out_shape)

        if 'out_mask' not in locals():
            out_mask = np.memmap(outfile, dtype='uint8', mode='r', shape=out_shape)

        #=======================================================
        ##step 5: write out masked dem and mask to file
    except:
        print("Error occurred with creating mask for file %s" % (dem))

    # try:
        # # get the profile of the geotif which contains info we need to write out rasters with the same CRS, dtype, etc
        # with rasterio.open(dem) as src: #rgb
            # profile = src.profile

        # #=============================
        # ##conf
        # if CALC_CONF:
            # if verbose:
                # print('.....................................')
                # print('Writing out mask confidence raster ...')
                # print(datetime.now())

            # profile['dtype'] = 'float32'
            # profile['nodata'] = 0.0
            # profile['count'] = 1

            # with rasterio.Env():
                # with rasterio.open(dem.replace('.tif','_automask_conf.tif'), 'w', **profile) as dst:
                    # dst.write(np.expand_dims(out_conf.astype('float32'),0))
            # del out_conf

    # except:
        # print("Error occurred with creating masked confidence file for %s" % (dem))

    try:
        #=============================
        ## dem
        if verbose:
            print('.....................................')
            print('Writing out masked DEM ...')
            print(datetime.now())

        with rasterio.open(dem) as src:
            profile = src.profile
            bigtiff = src.read()

        bigtiff = mask(bigtiff,profile,out_mask)
			
        with rasterio.Env():
            with rasterio.open(dem.replace('.tif','_automasked_dem.tif'), 'w', **profile) as dst:
                dst.write(bigtiff)
        del bigtiff

    except:
        print("Error occurred with creating masked dem file for %s" % (dem))

    try:
        #=============================
        ##mask
        if verbose:
            print('.....................................')
            print('Writing out mask raster ...')
            print(datetime.now())

        profile['nodata'] = 0.0
        profile['dtype'] = 'uint8'
        profile['count'] = 1

        with rasterio.Env():
            with rasterio.open(dem.replace('.tif','_automask.tif'), 'w', **profile) as dst:
                dst.write(np.expand_dims(out_mask,0))

    except:
        print("Error occurred with creating mask file for %s" % (dem))

    try:
        #=======================================================
        ##step 6: clean up temp files
        if 'outfile' in locals():
            outfile.close()
        if 'outfile2' in locals():
            outfile2.close()
        del out_mask

        if verbose:
            print("Program finished at %s" % datetime.now())
            # if os.name=='posix': # true if linux/mac
            elapsed = (time.time() - start)/60
            # else: # windows
            #    elapsed = (time.clock() - start)/60
            print("Processing took "+ str(elapsed) + " minutes")

    except:
        print("Error")


###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:m:o:f:t:")
    except getopt.GetoptError:
        print('python mask_dem.py -m model mode -o overlap percentage -f median filter radius -t otsu threshold \n')
        print('Defaults:\n')
        print('-m  = 1 (dem only - the other model modes are not implemented)\n')
        print('-o = 25 (percentage overlap between 5 and 95)\n')
        print('-f = 0 (median filter radius)\n')
        print('-t​​​​​​​ = 0 (Not otsu)\n')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage python mask_dem.py') 
            print('Example usage python mask_dem.py -m 1 -o 25 -f 5 -t 1') 
            print('Example usage: python mask_dem.py -m 2 -o 10 -f 10 -t 0') 
            sys.exit()

        elif opt in ("-m"):
            model_mode = arg
            model_mode = int(model_mode)
        elif opt in ("-o"):
            overlap = arg
            overlap = int(overlap)
        elif opt in ("-f"):
            medfilt_radius = arg
            medfilt_radius = int(medfilt_radius)
        elif opt in ("-t"):
            use_otsu = arg
            use_otsu = int(use_otsu)
			
    if 'model_mode' not in locals():
        model_mode = 1
    print(model_mode)
	
    if 'overlap' not in locals():
        overlap = 25 #25% overlap
    elif overlap<5:
        overlap = 5
        print("Overlap minimum of 5% imposed")
    elif overlap>95:
        overlap = 95
        print("Overlap maximum of 95% imposed")


    print("Overlap: %i"% (overlap)	)

    if 'medfilt_radius' not in locals():
        medfilt_radius = 0
	
    print("Median filter radius: %i" % (medfilt_radius))
	
    if 'use_otsu' not in locals():
        use_otsu = False
    elif type(use_otsu) is int:
        use_otsu = bool(use_otsu)
    print(use_otsu)	
	
	
    #====================================================
    print('.....................................')
    print('.....................................')
    print('.....................................')
    print('.....................................')
    print('.....................................')
    print('................A.D.M................')
    print('.....................................')
    print('.....................................')
    print('.....................................')
    print('.....................................')
    print('.....................................')

    # model_mode = 1
    #model_mode = 2
    #model_mode = 3
    #1=dem, 2=stdev, 3=rgb, 4=rgb+dem, 5=rgb+stdev

    if model_mode<3:
        root = Tk()
        dems = filedialog.askopenfilenames(initialdir = "./",title = "Select DEM file",filetypes = (("DSM/DEM geotiff file","*.tif"),("all files","*.*")))
        root.withdraw()
        print("%s files selected" % (len(dems)))
        rgbs = [None for k in dems]
    elif model_mode==3:
        root = Tk()
        dems = filedialog.askopenfilenames(initialdir = "./",title = "Select Orthomosaic file",filetypes = (("RGB Ortho geotiff file","*.tif"),("all files","*.*")))
        root.withdraw()
        print("%s files selected" % (len(dems)))
        rgbs = [None for k in dems]
    elif model_mode>3:
        root = Tk()
        rgbs = filedialog.askopenfilenames(initialdir = "./",title = "Select Orthomosaic file",filetypes = (("RGB Ortho geotiff file","*.tif"),("all files","*.*")))
        root.withdraw()
        print("%s files selected" % (len(rgbs)))
        rgbs = sorted(rgbs)

        root = Tk()
        dems = filedialog.askopenfilenames(initialdir = "./",title = "Select corresponding DEM file",filetypes = (("DSM/DEM geotiff file","*.tif"),("all files","*.*")))
        root.withdraw()
        print("%s files selected" % (len(dems)))
        dems = sorted(dems)

    try:
        for dem,rgb in zip(dems,rgbs):
            main(dem, rgb, model_mode, overlap, medfilt_radius, use_otsu) 
    except:
        print("Unspecified error. Check inputs. Exiting ...")
        sys.exit(2)


	# dem = r'F:/dbuscombe_github/dems/noisy/clip_20181006_Ophelia_Inlet_to_Beaufort_Inlet_1m_UTM18N_NAVD88_cog.tif'
	# dem = os.path.normpath(dem)




