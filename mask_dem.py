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
        bigtiff1[out_mask==0] = profile['nodata']
        bigtiff[0,:,:] = bigtiff1
        del bigtiff1

        bigtiff2 = bigtiff[1,:,:]
        bigtiff2[out_mask==0] = profile['nodata']
        bigtiff[1,:,:] = bigtiff2
        del bigtiff2

        bigtiff3 = bigtiff[2,:,:]
        bigtiff3[out_mask==0] = profile['nodata']
        bigtiff[2,:,:] = bigtiff3
        del bigtiff3

        bigtiff4 = bigtiff[3,:,:]
        bigtiff4[out_mask==0] = profile['nodata']
        bigtiff[3,:,:] = bigtiff4
        del bigtiff4
    else:
        bigtiff1 = bigtiff[0,:,:]
        bigtiff1[out_mask==0] = profile['nodata']
        bigtiff[0,:,:] = bigtiff1
        del bigtiff1

    return bigtiff

#-----------------------------------
def main(dem, model_mode): #threshold,
    """
    main program
    """
    print('.....................................')
    print('.....................................')
    print(datetime.now())
    print("Working on %s" % (dem))
    tilesize = 1024 #
    verbose = True

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

    do_plot=True #False #True
    if do_plot:
        import matplotlib.pyplot as plt

    if verbose:
        # start timer
        if os.name=='posix': # true if linux/mac or cygwin on windows
           start = time.time()
        else: # windows
           start = time.clock()

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
        #pre-allocate large arrays to be eventually written to geotiff format files, n, out_mask and out_conf
        #each are padded to accomodate half a window length overlap
        # out_mask = np.zeros((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.float16)
        # if CALC_CONF:
        #     out_conf = np.ones((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.float32)
        # n = np.zeros((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.uint8)

        out_mask = np.zeros((padwidth, padheight), dtype=np.uint8)
        if CALC_CONF:
            out_conf = np.ones((padwidth, padheight), dtype=np.float32)
        #n = np.zeros((padwidth, padheight), dtype=np.uint8)
        # i = 4096
        # j = 9216
        i = 2048 ; j = 10240

        for i in tqdm(range(0, padwidth, tilesize)):
            for j in range(0, padheight, tilesize):
                window = rasterio.windows.Window(i,j,tilesize, tilesize)

                with rasterio.open(dem) as src:
                    profile = src.profile
                    orig = src.read(window=window).squeeze()

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

                    if do_plot:
                        plt.subplot(221); plt.imshow(image, cmap='gray'); plt.axis('off'); plt.colorbar(shrink=.5)

                    if image.shape[0] < tilesize:

                        if model_mode<3:
                            subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
                            x,y=image.shape
                            subset_pad[:x,:y] = image
                        elif model_mode==3:
                            x,y,z=image.shape
                            subset_pad = np.zeros((tilesize, tilesize,z), dtype=profile['dtype'])
                            subset_pad[:x,:y,:z] = image

                        del image
                        image = subset_pad.copy()
                        del subset_pad

                    elif image.shape[1] < tilesize:
                        if model_mode<3:
                            subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
                            x,y=image.shape
                            subset_pad[:x,:y] = image
                        elif model_mode==3:
                            x,y,z=image.shape
                            subset_pad = np.zeros((tilesize, tilesize,z), dtype=profile['dtype'])
                            subset_pad[:x,:y,:z] = image
                        del image
                        image = subset_pad.copy()#/255
                        del subset_pad

                    image = tf.expand_dims(image, 0)
                    # use model in prediction mode
                    est_label = model.predict(image , batch_size=1).squeeze()
                    image = image.numpy().squeeze()

                    est_label = medianfiltmask(est_label, radius=21)/255.

                    conf = np.std(est_label, -1)

                    # est_label = np.argmax(est_label, -1) #2=bad, 1=good, 0=bad
                    # est_label = (est_label==1).astype('uint8') #only good

                    est_label_orig = est_label.copy()
                    est_label = np.zeros((est_label.shape[0], est_label.shape[1])) #np.argmax(est_label, -1).astype(np.uint8).T
                    est_label[est_label_orig[:,:,0]>.5] = 1  #nodata
                    est_label[est_label_orig[:,:,1]>.5] = 1  #good
                    est_label[est_label_orig[:,:,2]>.25] = 0  #bad
                    est_label[est_label>1] = 1

                    if do_plot:
                        plt.subplot(222); plt.imshow(est_label, cmap='bwr'); plt.axis('off'); plt.colorbar(shrink=.5)
                        plt.subplot(223); plt.imshow(image, cmap='gray'); plt.imshow(est_label, cmap='bwr', alpha=0.3); plt.axis('off'); plt.colorbar(shrink=.5)

                    #confidence is computed as deviation from 0.5
                    #conf = np.abs(est_label-0.5)/2#1-est_label
                    #conf[est_label<.5] = est_label[est_label<.5]
                    #conf = 1-conf

                    if do_plot:
                        plt.subplot(224); plt.imshow(conf, cmap='Blues'); plt.axis('off'); plt.colorbar(shrink=.5)
                        plt.savefig('aom_ex'+str(i)+'_'+str(j)+'.png', dpi=300); plt.close()

                    #fill out big rasters
                    out_mask[i:i+tilesize,j:j+tilesize] += est_label.astype('uint8') #fill in that portion of big mask
                    del est_label
                    if CALC_CONF:
                        out_conf[i:i+tilesize,j:j+tilesize] += conf.astype('float32') #fill out that portion of the big confidence raster
                        del conf
                    #n[i:i+tilesize,j:j+tilesize] += 1

        #=======================================================
        ##step 4: crop, rotate arrays, divide by N, and write to memory mapped temporary file
        #(i know, classy)
        out_mask = out_mask[:width,:height]
        out_mask = np.rot90(np.fliplr(out_mask))
        out_shape = out_mask.shape

        # n = n[:width,:height]
        # n = np.rot90(np.fliplr(n))
        out_conf = out_conf[:width,:height]
        out_conf = np.rot90(np.fliplr(out_conf))

        # out_mask = np.divide(out_mask, n, out=out_mask, casting='unsafe' )

        # if threshold == 'otsu':
        #     out_mask[np.isnan(out_mask)]=0
        #     thres = threshold_otsu(out_mask)
        #     print("Otsu threshold: %f" % (thres))
        # else:
        #     thres = threshold

        out_mask = out_mask.astype('uint8')
        out_mask = (out_mask!=1).astype('uint8')
        #
        #
        # out_mask[out_mask<0.5] = 0
        # out_mask[out_mask<=1] = 0
        # out_mask[out_mask>=1] = 1

        if CALC_CONF:
            # out_conf = np.divide(out_conf, np.maximum(0,n.astype('float')), out=out_conf, casting='unsafe')
            out_conf[out_conf==np.nan]=0
            out_conf[out_conf==np.inf]=0
            out_conf[out_conf<=0] = 0
            out_conf = out_conf-1
            #out_conf[out_conf>.5] = 0
            # out_conf[out_conf==-1] = 0
            #out_mask = np.memmap(outfile, dtype='uint8', mode='r', shape=out_shape)
            out_conf[out_mask==0] = 0

        # out_mask[out_conf<.25] = 0

        outfile = TemporaryFile()
        fp = np.memmap(outfile, dtype='uint8', mode='w+', shape=out_mask.shape)
        fp[:] = out_mask[:]
        fp.flush()
        del out_mask
        del fp


        #write out to temporary memory mapped file
        # del n
        outfile2 = TemporaryFile()
        fp = np.memmap(outfile2, dtype='float32', mode='w+', shape=out_conf.shape)
        fp[:] = out_conf[:]
        fp.flush()
        del out_conf
        del fp

        #read back in again
        out_conf = np.memmap(outfile2, dtype='float32', mode='r', shape=out_shape)

        if 'out_mask' not in locals():
            out_mask = np.memmap(outfile, dtype='uint8', mode='r', shape=out_shape)

        #=======================================================
        ##step 5: write out masked dem and mask to file
    except:
        print("Error occurred with creating mask for file %s" % (dem))

    try:
        # get the profile of the geotif which contains info we need to write out rasters with the same CRS, dtype, etc
        with rasterio.open(dem) as src: #rgb
            profile = src.profile

        #=============================
        ##conf
        if CALC_CONF:
            if verbose:
                print('.....................................')
                print('Writing out mask confidence raster ...')
                print(datetime.now())

            profile['dtype'] = 'float32'
            profile['nodata'] = 0.0
            profile['count'] = 1

            with rasterio.Env():
                with rasterio.open(dem.replace('.tif','_automask_conf.tif'), 'w', **profile) as dst:
                    dst.write(np.expand_dims(out_conf.astype('float32'),0))
            del out_conf

    except:
        print("Error occurred with creating masked confidence file for %s" % (dem))

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

            if os.name=='posix': # true if linux/mac
               elapsed = (time.time() - start)/60
            else: # windows
               elapsed = (time.clock() - start)/60
            print("Processing took "+ str(elapsed) + " minutes")

    except:
        print("Error")


###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:m:t:") #t:
    except getopt.GetoptError:
        print('python mask_dem.py -m mode -t dataType') #
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage (defaults = prompt): python mask_dem.py') #, threshold=0.5
            print('Example usage (autobatch): python mask_dem.py -m autobatch ') #, and threshold=0.6 -t 0.6
            print('Example usage: python mask_dem.py -m prompt') # (otsu threshold)  -t otsu
            sys.exit()
        elif opt in ("-m"):
            mode = arg
        elif opt in ("-t"):
            model_mode = arg
            model_mode = int(model_mode)

    if 'mode' not in locals():
        mode = 'prompt'

    if 'model_mode' not in locals():
        model_mode = 1
    print(model_mode)
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

    if mode =='autobatch':
        try:
            for dem in glob('*.tif'):
                main(dem, model_mode) #threshold,
        except:
            print("No tif files found - check inputs for 'autobatch' or use 'prompt' mode. Exiting ...")
            sys.exit(2)
    else: #'prompt' is default

        if model_mode<3:
            root = Tk()
            dems = filedialog.askopenfilenames(initialdir = "./",title = "Select file",filetypes = (("DSM/DEM geotiff file","*.tif"),("all files","*.*")))
            root.withdraw()
            print("%s files selected" % (len(dems)))
        elif model_mode==3:
            root = Tk()
            dems = filedialog.askopenfilenames(initialdir = "./",title = "Select file",filetypes = (("RGB Ortho geotiff file","*.tif"),("all files","*.*")))
            root.withdraw()
            print("%s files selected" % (len(dems)))

        try:
            for dem in dems:
                main(dem, model_mode) #threshold,
        except:
            print("Unspecified error. Check inputs. Exiting ...")
            sys.exit(2)


                #orig = orig.squeeze()
                    #image = image/255

        #main(dem)

    #TARGET_SIZE = [tilesize, tilesize]
    #BATCH_SIZE = 6
    #NCLASSES = 1

    # if DO_PARALLEL:
    #     from joblib import Parallel, delayed

        # counter = 0
        # out_mask = np.zeros((padwidth+tilesize, padheight+tilesize), dtype=np.float16)
        # if CALC_CONF:
        #     out_conf = np.ones((padwidth+tilesize, padheight+tilesize), dtype=np.float32)
        # n = np.zeros((padwidth+tilesize, padheight+tilesize), dtype=np.uint8)
        # for i in tqdm(range(0, padwidth, tilesize)):
        #     for j in range(0, padheight, tilesize):

    # if USE_RGB:
    #     weights_path = 'orthoclip_2class_batch_6.h5'
    #     json_file = open(weights_path.replace('.h5','.json'), 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     model = model_from_json(loaded_model_json)
    # else:
# if USE_RGB:
#     root = Tk()
#     root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("ortho geotiff file","*.tif"),("all files","*.*")))
#     rgb = root.filename
#     root.withdraw()
#     print("... and %s" % (rgb))
#
# #-----------------------------------
# def mean_iou(y_true, y_pred):
#     """
#     mean_iou(y_true, y_pred)
#     This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions
#
#     INPUTS:
#         * y_true: true masks, one-hot encoded.
#             * Inputs are B*W*H*N tensors, with
#                 B = batch size,
#                 W = width,
#                 H = height,
#                 N = number of classes
#         * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
#             * Inputs are B*W*H*N tensors, with
#                 B = batch size,
#                 W = width,
#                 H = height,
#                 N = number of classes
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * IoU score [tensor]
#     """
#     yt0 = y_true[:,:,:,0]
#     yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
#     inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
#     union = tf.math.count_nonzero(tf.add(yt0, yp0))
#     iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
#     return iou
#
# #-----------------------------------
# def dice_coef(y_true, y_pred):
#     """
#     dice_coef(y_true, y_pred)
#
#     This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions
#
#     INPUTS:
#         * y_true: true masks, one-hot encoded.
#             * Inputs are B*W*H*N tensors, with
#                 B = batch size,
#                 W = width,
#                 H = height,
#                 N = number of classes
#         * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
#             * Inputs are B*W*H*N tensors, with
#                 B = batch size,
#                 W = width,
#                 H = height,
#                 N = number of classes
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * Dice score [tensor]
#     """
#     smooth = 1.
#     y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
#     y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


    #out_conf = out_conf/100
    #out_conf = out_conf.astype('float32')
    #out_conf = da.floor_divide(out_conf, n+0.00001)
    #out_conf = out_conf.astype('float32')
    #out_conf[out_conf>=1] = 0
    #out_conf[out_conf<.5] = 0
    #out_conf[out_conf>=.9] = 0
    #out_conf[out_conf<.5] = 1
    # out_conf = out_conf[:width,:height]
    # out_conf = np.rot90(np.fliplr(out_conf))

# prefix = id_generator()+'_' #'example_'
# #-----------------------------------
# def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
#     """
#     generate a random string
#     """
#     return ''.join(random.choice(chars) for _ in range(size))
# from skimage.io import imsave, imread
# from skimage.transform import resize
# import itertools #, random, string#, gc

#
# print("Creating dask arrays %s" % datetime.now())
# n = da.from_array(n, chunks = 'auto') #chunks=(1000, 1000))
#
# out_mask = da.from_array(out_mask, chunks='auto') #(1000, 1000))
# if CALC_CONF:
#     out_conf = da.from_array(out_conf, chunks='auto') #(1000, 1000))
#
# out_mask = da.floor_divide(out_mask, n)

# out_mask = np.divide(out_mask, n+0.00001) #out_conf/n


# if USE_RGB:
#     model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 4), BATCH_SIZE, NCLASSES)
# else:
#     model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 1), BATCH_SIZE, NCLASSES)
#
#

# if USE_RGB:
#     weights = 'orthoclip_2class_batch_6.h5'
# else:
#     weights = 'orthoclip_stdevonly_2class_batch_6.h5'

# if USE_RGB:
#     model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 4), BATCH_SIZE, NCLASSES)
# else:
#     model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 1), BATCH_SIZE, NCLASSES)
#
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
#
# model.load_weights(weights)
# #
# model_json = model.to_json()
# with open(weights.replace('.h5','.json'), "w") as json_file:
#     json_file.write(model_json)


# import dask_image.imread
# import dask.delayed as dd
# from dask.distributed import Client

### step 1 : make chunks and save them to temporary png files
# print('.....................................')
# print('Making small image chunks ...')
# print(datetime.now())


# if OVERLAP:
# out_mask = np.memmap(outfile, dtype='uint8', mode='r', shape=out_shape)

#
# #out_mask[out_mask==np.nan]=0
# out_mask = np.divide(out_mask, n, out=out_mask)#, casting='unsafe' )# out_mask/n #divide out by number of times each cell was sampled

# out_mask[out_mask>1] = 1

####outfile = os.path.join(mkdtemp(), 'mask.dat')

# gc.collect()

#
# #-----------------------------------
# def chunk_image(i,j,tilesize, dem, USE_RGB, prefix, counter, rgb=None):
#     """
#     create and save image chunks to file
#     """
#     #print((counter,i,j))
#     window = rasterio.windows.Window(i,j,tilesize, tilesize)
#
#     with rasterio.open(dem) as src: #rgb
#         profile = src.profile
#         subset = src.read(window=window)
#     # print((i,j))
#     #print(np.max(subset))
#
#     orig = subset.squeeze()
#     if len(np.unique(orig))>1:
#         orig[orig==profile['nodata']] = 0
#         subset = std_convoluted(orig, 3)
#         subset = z_exp(subset,n=0.5)
#         #print(np.max(subset))
#         subset = np.squeeze(subset).T
#     else:
#         subset=np.zeros((tilesize, tilesize), dtype=profile['dtype'])
#     del orig
#
#     if subset.shape[0] < tilesize:
#         #print('Padding %i' % (counter))
#         subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
#         x,y=subset.shape
#         subset_pad[:x,:y] = subset
#         del subset
#         subset = subset_pad.copy()
#     elif subset.shape[1] < tilesize:
#         #print('Padding %i' % (counter))
#         subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
#         x,y=subset.shape
#         subset_pad[:x,:y] = subset
#         del subset
#         subset = subset_pad.copy()
#         del subset_pad
#
#     #print('size: %i %i'% (subset.shape[0], subset.shape[1]))
#
#     if counter<10:
#         imsave('tmp/stdev/'+prefix+'000000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
#     elif counter<100:
#         imsave('tmp/stdev/'+prefix+'00000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
#     elif counter<1000:
#         imsave('tmp/stdev/'+prefix+'0000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
#     elif counter<10000:
#         imsave('tmp/stdev/'+prefix+'000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
#     else:
#         imsave('tmp/stdev/'+prefix+'00'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
#
#     del subset
#
#     if USE_RGB:
#         with rasterio.open(rgb) as src: #rgb
#             profile = src.profile
#             subset = src.read(window=window)
#
#         subset = np.squeeze(subset).T
#         # print('size: %i %i'% (subset.shape[0], subset.shape[1]))
#
#         if subset.shape[0] < tilesize:
#             #print('Padding %i' % (counter))
#             subset_pad = np.zeros((tilesize, tilesize, subset.shape[-1]), dtype=profile['dtype'])
#             x,y,z=subset.shape
#             subset_pad[:x,:y,:z] = subset
#             del subset
#             subset = subset_pad.copy()
#         elif subset.shape[1] < tilesize:
#             #print('Padding %i' % (counter))
#             subset_pad = np.zeros((tilesize, tilesize, subset.shape[-1]), dtype=profile['dtype'])
#             x,y,z=subset.shape
#             subset_pad[:x,:y,:z] = subset
#             del subset
#             subset = subset_pad.copy()
#             del subset_pad
#
#         #print('size: %i %i'% (subset.shape[0], subset.shape[1]))
#
#         if counter<10:
#             imsave('tmp/ortho/'+prefix+'000000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
#         elif counter<100:
#             imsave('tmp/ortho/'+prefix+'00000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
#         elif counter<1000:
#             imsave('tmp/ortho/'+prefix+'0000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
#         elif counter<10000:
#             imsave('tmp/ortho/'+prefix+'000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
#         else:
#             imsave('tmp/ortho/'+prefix+'00'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
#
#         del subset

# #-----------------------------------
# def mask_chunk(sample_filename, tilesize):
#     """
#     create mask from image chunk
#     """
#     threshold = 0.5
#
#     if USE_RGB:
#         image = seg_file2tensor_4band(sample_filename.replace('stdev', 'ortho'), sample_filename )/255
#     else:
#         image = seg_file2tensor_1band(sample_filename)/255
#
#     if len(np.unique(image))>1:
#         image = tf.expand_dims(image, 0)
#         #est_label = model.predict(image , batch_size=1).squeeze()
#
#         R = []; W = []
#         counter = 0
#         for k in np.linspace(0,int(image.shape[0]/5),5):
#             k = int(k)
#             R.append(np.roll(model.predict(np.roll(image,k) , batch_size=1).squeeze(),-k))
#             counter +=1
#             if k==0:
#                 W.append(0.1)
#             else:
#                 W.append(1/np.sqrt(k))
#
#         K.clear_session()
#         est_label = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
#
#         #confidence is computed as deviation from 0.5
#         conf = 1-est_label
#         conf[est_label<threshold] = est_label[est_label<threshold]
#         conf = 1-conf
#         #model_conf = np.sum(conf)/np.prod(conf.shape)
#         #print('Overall model confidence = %f'%(model_conf))
#
#         est_label[est_label<threshold] = 0
#         est_label[est_label>threshold] = 1
#         #conf = (100*conf).astype('uint8')  #turn this into 8-bit for more efficient ram usage
#         # conf[conf<50]=50
#         # conf[conf>100]=100
#
#     else:
#         est_label = conf = None
#
#     return est_label, conf

# try:
#     os.mkdir('tmp')
#     if USE_RGB:
#         os.mkdir('tmp/ortho')
#     os.mkdir('tmp/stdev')
# except:
#     pass
#
# if DO_PARALLEL:
#     if OVERLAP:
#         ivec = range(0, padwidth, int(tilesize/2)) #25% overlap
#         jvec = range(0, padheight, int(tilesize/2))
#     else:
#         ivec = range(0, padwidth, tilesize)
#         jvec = range(0, padheight, tilesize)
#
#     #w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(print)((counter,i,j)) for counter,(i,j) in enumerate(itertools.product(ivec,jvec)))
#     if USE_RGB:
#         w = Parallel(n_jobs=-1, verbose=0)(delayed(chunk_image)(i,j,tilesize, dem, USE_RGB, prefix, counter, rgb) for counter,(i,j) in enumerate(itertools.product(ivec,jvec)))
#     else:
#         w = Parallel(n_jobs=-1, verbose=0)(delayed(chunk_image)(i,j,tilesize, dem, USE_RGB, prefix, counter) for counter,(i,j) in enumerate(itertools.product(ivec,jvec)))
#
# else:
#     counter=0
#     for i in tqdm(range(0, padwidth, tilesize)):
#         for j in range(0, padheight, tilesize):
#
#             if USE_RGB:
#                 chunk_image(i,j,tilesize, dem, USE_RGB, prefix, counter, rgb)
#             else:
#                 chunk_image(i,j,tilesize, dem, USE_RGB, prefix, counter)
#
#             counter+=1


#=======================================================
## step 3: use the model for prediction


# sample_filenames = sorted(tf.io.gfile.glob('tmp/stdev'+os.sep+'*.png'))
# if OVERLAP:
#     print('Number of samples: %i (50 percent overlap)' % (len(sample_filenames)))
# else:
#     print('Number of samples: %i' % (len(sample_filenames)))

# def apply_mask(image):
#     # return model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
#
#     if np.max(image).compute()<1:
#         return np.zeros((TARGET_SIZE[0], TARGET_SIZE[1]), dtype=np.float32), np.zeros((TARGET_SIZE[0], TARGET_SIZE[1]), dtype=np.float32)
#     else:
#
#         if USE_RGB:
#             model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 4), BATCH_SIZE, NCLASSES)
#         else:
#             model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 1), BATCH_SIZE, NCLASSES)
#
#         model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
#
#         model.load_weights(weights)
#         K.clear_session()
#
#         est_label = model.predict(image.compute().numpy().T , batch_size=1).squeeze()
#         K.clear_session()
#
#         conf = 1-est_label
#         conf[est_label<.5] = est_label[est_label<.5]
#         conf = 1-conf
#
#         return est_label, conf
#

#
# filename_pattern ='tmp/stdev'+os.sep+'*.png'
# # x = dask_image.imread.imread(sample_filenames)
#
# lazy_arrays = [dd(seg_file2tensor_1band)(fn) for fn in sorted(glob(filename_pattern))]
# # Create a dask array from a list of chunk png files as a dask delayed value
# lazy_arrays = [da.from_delayed(x, shape=(tilesize, tilesize), dtype=np.uint8) for x in tqdm(lazy_arrays)]
# lazy_arrays = da.stack(lazy_arrays) #we stack the array to make it 4d (2d x depth) rather than default
# print("Array of size {} imported".format(lazy_arrays.shape))

# i never understand dask ...
# client = Client()  # set up local cluster on your laptop
# client
#result = client.submit(apply_mask, lazy_arrays)
# w = apply_mask(lazy_arrays)

# w = Parallel(n_jobs=-1, verbose=0, max_nbytes=None)(delayed(apply_mask)(lazy_arrays[k]) for k in tqdm(range(len(lazy_arrays)))) #pre_dispatch='2 * n_jobs',

#
# counter=0
# if OVERLAP:
#     out_mask = np.zeros((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.uint8)
#     if CALC_CONF:
#         out_conf = np.ones((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.float32)
#     n = np.zeros((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.uint8)
#     for i in tqdm(range(0, width, int(tilesize/2))):
#         for j in range(0, height, int(tilesize/2)):
#
#             est_label, conf = mask_chunk(sample_filenames[counter], tilesize)
#             #est_label[np.isnan(est_label)] = 0
#             #est_label[np.isinf(est_label)] = 0
#             if conf is not None:
#                 out_mask[i:i+tilesize,j:j+tilesize] += est_label.astype('uint8') #fill in that portion of big mask
#                 if CALC_CONF:
#                     out_conf[i:i+tilesize,j:j+tilesize] += conf.astype('float32') #fill out that portion of the big confidence raster
#                 n[i:i+tilesize,j:j+tilesize] += 1
#
#             counter +=1
#             if CALC_CONF:
#                 del conf
#             del est_label
#     print(datetime.now())
#     #out_mask[out_mask==np.nan]=0
#     out_mask = np.divide(out_mask, n, out=out_mask, casting='unsafe' )# out_mask/n #divide out by number of times each cell was sampled
#     out_mask = out_mask.astype('uint8')
#     out_mask[out_mask>1] = 1
#
#     out_mask = out_mask[:width,:height]
#     out_mask = np.rot90(np.fliplr(out_mask))
#     out_shape = out_mask.shape
#
#     #outfile = os.path.join(mkdtemp(), 'mask.dat')
#     outfile = TemporaryFile()
#     fp = np.memmap(outfile, dtype='uint8', mode='w+', shape=out_mask.shape)
#     fp[:] = out_mask[:]
#     fp.flush()
#     del out_mask
#     del fp
#     gc.collect()
#
#     if CALC_CONF:
#         #out_conf = out_conf/100
#         out_conf = out_conf.astype('float32')
#         out_conf = np.divide(out_conf, n+0.00001) #out_conf/n
#         del n
#         out_conf = out_conf.astype('float32')
#         out_conf[out_conf==np.nan]=0
#         out_conf[out_conf==np.inf]=0
#         out_conf[out_conf>=1] = 0
#         out_conf[out_conf<.5] = 0
#         out_conf[out_conf>=.9] = 0
#         out_conf[out_conf<.5] = 1
#         out_conf = out_conf[:width,:height]
#         out_conf = np.rot90(np.fliplr(out_conf))
#
# else:
#
#     out_mask = np.zeros((padwidth, padheight), dtype=np.uint8)
#     if CALC_CONF:
#         out_conf = np.ones((padwidth, padheight), dtype=np.uint8)
#     for i in tqdm(range(0, width, tilesize)):
#         for j in range(0, height, tilesize):
#
#             est_label, conf = mask_chunk(sample_filenames[counter], tilesize)
#             if conf is not None:
#                 out_mask[i:i+tilesize,j:j+tilesize] = est_label.astype('uint8') #fill in that portion of big mask
#                 if CALC_CONF:
#                     out_conf[i:i+tilesize,j:j+tilesize] = conf.astype('uint8') #fill out that portion of the big confidence raster
#
#             counter +=1
#             if CALC_CONF:
#                 del conf
#             del est_label
#
#     out_mask = out_mask[:width,:height]
#     out_mask = np.rot90(np.fliplr(out_mask))
#     out_shape = out_mask.shape
#
#     if CALC_CONF:
#         out_conf = out_conf.astype('float32')
#         out_conf = out_conf[:width,:height]
#         out_conf = np.rot90(np.fliplr(out_conf))


# out_conf = out_conf.astype('float32')
# out_mask = out_mask.astype('uint8')


#=======================================================
# step 4, tify up
# delete tmp files
# print('.....................................')
# print('Removing temporary images ...')
#
# if USE_RGB:
#     for f in glob('tmp/ortho/*.*'):
#         os.remove(f)
#     os.rmdir('tmp/ortho')
# for f in glob('tmp/stdev/*.*'):
#     os.remove(f)
# os.rmdir('tmp/stdev')
# os.rmdir('tmp')
# gc.collect()


# #=============================
# # mask orhto
# if USE_RGB:
#     with rasterio.open(rgb) as src: #rgb
#         profile = src.profile
#         bigtiff = src.read()
#
#     bigtiff = mask(bigtiff,profile,out_mask)
#     del out_mask
#
#     #wrte maked geotiff to file
#     with rasterio.Env():
#         with rasterio.open(rgb.replace('.tif','_automasked_rgb.tif'), 'w', **profile) as dst:
#             dst.write(bigtiff)
#     del bigtiff
#     gc.collect()
#
# else:
#     from tkinter import messagebox
#     root= Tk()
#     MsgBox = messagebox.askquestion ('Mask ortho?','Would you like to use this mask to mask an ortho?',icon = 'warning')
#     root.destroy()
#     if MsgBox == 'yes':
#         root = Tk()
#         root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("ortho geotiff file","*.tif"),("all files","*.*")))
#         rgb = root.filename
#         root.withdraw()
#
#         print('.....................................')
#         print('Writing out masked RGB ortho ...')
#         print(datetime.now())
#
#         with rasterio.open(rgb) as src: #rgb
#             profile = src.profile
#             bigtiff = src.read()
#
#         bigtiff = mask(bigtiff,profile,out_mask)
#         del out_mask
#
#         #wrte maked geotiff to file
#         with rasterio.Env():
#             with rasterio.open(rgb.replace('.tif','_automasked_rgb.tif'), 'w', **profile) as dst:
#                 dst.write(bigtiff)
#         del bigtiff
#         gc.collect()

# #-----------------------------------
# def seg_file2tensor_1band(f):
#     """
#     "seg_file2tensor(f)"
#     This function reads a jpeg image from file into a cropped and resized tensor,
#     for use in prediction with a trained segmentation model
#     INPUTS:
#         * f [string] file name of jpeg
#     OPTIONAL INPUTS: None
#     OUTPUTS:
#         * image [tensor array]: unstandardized image
#     GLOBAL INPUTS: TARGET_SIZE
#     """
#     bits = tf.io.read_file(f)
#     if 'jpg' in f:
#         image = tf.image.decode_jpeg(bits)
#     elif 'png' in f:
#         image = tf.image.decode_png(bits)
#
#     w = tf.shape(image)[0]
#     h = tf.shape(image)[1]
#     tw = TARGET_SIZE[0]
#     th = TARGET_SIZE[1]
#     resize_crit = (w * th) / (h * tw)
#     image = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
#                  )
#
#     nw = tf.shape(image)[0]
#     nh = tf.shape(image)[1]
#     image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#     return image
#
# #-----------------------------------
# def seg_file2tensor_4band(f, fir):
#     """
#     "seg_file2tensor(f)"
#     This function reads a jpeg image from file into a cropped and resized tensor,
#     for use in prediction with a trained segmentation model
#     INPUTS:
#         * f [string] file name of jpeg
#     OPTIONAL INPUTS: None
#     OUTPUTS:
#         * image [tensor array]: unstandardized image
#     GLOBAL INPUTS: TARGET_SIZE
#     """
#     bits = tf.io.read_file(f)
#     if 'jpg' in f:
#         image = tf.image.decode_jpeg(bits)
#     elif 'png' in f:
#         image = tf.image.decode_png(bits)
#
#     bits = tf.io.read_file(fir)
#     if 'jpg' in fir:
#         nir = tf.image.decode_jpeg(bits)
#     elif 'png' in f:
#         nir = tf.image.decode_png(bits)
#
#     image = tf.concat([image, nir],-1)[:,:,:4]
#
#     w = tf.shape(image)[0]
#     h = tf.shape(image)[1]
#     tw = TARGET_SIZE[0]
#     th = TARGET_SIZE[1]
#     resize_crit = (w * th) / (h * tw)
#     image = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
#                  )
#
#     nw = tf.shape(image)[0]
#     nh = tf.shape(image)[1]
#     image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#     return image
#
