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
from skimage.io import imsave, imread
import os, itertools, random, string, gc
from scipy.signal import convolve2d
from tkinter import filedialog
from tkinter import *
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
from tempfile import mkdtemp

# =========================================================
USE_GPU = True

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf #numerical operations on gpu
import tensorflow.keras.backend as K

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# =========================================================
#-----------------------------------
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    generate a random string
    """
    return ''.join(random.choice(chars) for _ in range(size))

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

#-----------------------------------
def mean_iou(y_true, y_pred):
    """
    mean_iou(y_true, y_pred)
    This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * IoU score [tensor]
    """
    yt0 = y_true[:,:,:,0]
    yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

#-----------------------------------
def dice_coef(y_true, y_pred):
    """
    dice_coef(y_true, y_pred)

    This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice score [tensor]
    """
    smooth = 1.
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

#-----------------------------------
def batchnorm_act(x):
    """
    batchnorm_act(x)
    This function applies batch normalization to a keras model layer, `x`, then a relu activation function
    INPUTS:
        * `z` : keras model layer (should be the output of a convolution or an input layer)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * batch normalized and relu-activated `x`
    """
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)

#-----------------------------------
def conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
    """
    conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
    This function applies batch normalization to an input layer, then convolves with a 2D convol layer
    The two actions combined is called a convolutional block

    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`:input keras layer to be convolved by the block
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the batch normalized convolution
    """
    conv = batchnorm_act(x)
    return tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

#-----------------------------------
def bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
    """
    bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

    This function creates a bottleneck block layer, which is the addition of a convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between convolutional and bottleneck layers
    """
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([conv, bottleneck])

#-----------------------------------
def res_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
    """
    res_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

    This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])

#-----------------------------------
def upsamp_concat_block(x, xskip):
    """
    upsamp_concat_block(x, xskip)
    This function takes an input layer and creates a concatenation of an upsampled version and a residual or 'skip' connection
    INPUTS:
        * `xskip`: input keras layer (skip connection)
        * `x`: input keras layer
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    return tf.keras.layers.Concatenate()([u, xskip])

#-----------------------------------
def res_unet(sz, f, nclasses=1):
    """
    res_unet(sz, f, nclasses=1)
    This function creates a custom residual U-Net model for image segmentation
    INPUTS:
        * `sz`: [tuple] size of input image
        * `f`: [int] number of filters in the convolutional block
        * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
        * nclasses [int]: number of classes
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras model
    """
    inputs = tf.keras.layers.Input(sz)

    ## downsample
    e1 = bottleneck_block(inputs, f); f = int(f*2)
    e2 = res_block(e1, f, strides=2); f = int(f*2)
    e3 = res_block(e2, f, strides=2); f = int(f*2)
    e4 = res_block(e3, f, strides=2); f = int(f*2)
    _ = res_block(e4, f, strides=2)

    ## bottleneck
    b0 = conv_block(_, f, strides=1)
    _ = conv_block(b0, f, strides=1)

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f)

    ## classify
    if nclasses==1:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="sigmoid")(_)
    else:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="softmax")(_)

    #model creation
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

#-----------------------------------
def seg_file2tensor_1band(f):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bits = tf.io.read_file(f)
    if 'jpg' in f:
        image = tf.image.decode_jpeg(bits)
    elif 'png' in f:
        image = tf.image.decode_png(bits)

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE[0]
    th = TARGET_SIZE[1]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image

#-----------------------------------
def seg_file2tensor_4band(f, fir):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bits = tf.io.read_file(f)
    if 'jpg' in f:
        image = tf.image.decode_jpeg(bits)
    elif 'png' in f:
        image = tf.image.decode_png(bits)

    bits = tf.io.read_file(fir)
    if 'jpg' in fir:
        nir = tf.image.decode_jpeg(bits)
    elif 'png' in f:
        nir = tf.image.decode_png(bits)

    image = tf.concat([image, nir],-1)[:,:,:4]

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE[0]
    th = TARGET_SIZE[1]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image

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
def chunk_image(i,j,tilesize, dem, USE_RGB, prefix, counter, rgb=None):
    """
    create and save image chunks to file
    """
    #print((counter,i,j))
    window = rasterio.windows.Window(i,j,tilesize, tilesize)

    with rasterio.open(dem) as src: #rgb
        profile = src.profile
        subset = src.read(window=window)
    # print((i,j))
    #print(np.max(subset))

    orig = subset.squeeze()
    if len(np.unique(orig))>1:
        orig[orig==profile['nodata']] = 0
        subset = std_convoluted(orig, 3)
        subset = z_exp(subset,n=0.5)
        #print(np.max(subset))
        subset = np.squeeze(subset).T
    else:
        subset=np.zeros((tilesize, tilesize), dtype=profile['dtype'])
    del orig

    if subset.shape[0] < tilesize:
        #print('Padding %i' % (counter))
        subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
        x,y=subset.shape
        subset_pad[:x,:y] = subset
        del subset
        subset = subset_pad.copy()
    elif subset.shape[1] < tilesize:
        #print('Padding %i' % (counter))
        subset_pad = np.zeros((tilesize, tilesize), dtype=profile['dtype'])
        x,y=subset.shape
        subset_pad[:x,:y] = subset
        del subset
        subset = subset_pad.copy()
        del subset_pad

    #print('size: %i %i'% (subset.shape[0], subset.shape[1]))

    if counter<10:
        imsave('tmp/stdev/'+prefix+'000000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
    elif counter<100:
        imsave('tmp/stdev/'+prefix+'00000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
    elif counter<1000:
        imsave('tmp/stdev/'+prefix+'0000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
    elif counter<10000:
        imsave('tmp/stdev/'+prefix+'000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
    else:
        imsave('tmp/stdev/'+prefix+'00'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)

    del subset

    if USE_RGB:
        with rasterio.open(rgb) as src: #rgb
            profile = src.profile
            subset = src.read(window=window)

        subset = np.squeeze(subset).T
        # print('size: %i %i'% (subset.shape[0], subset.shape[1]))

        if subset.shape[0] < tilesize:
            #print('Padding %i' % (counter))
            subset_pad = np.zeros((tilesize, tilesize, subset.shape[-1]), dtype=profile['dtype'])
            x,y,z=subset.shape
            subset_pad[:x,:y,:z] = subset
            del subset
            subset = subset_pad.copy()
        elif subset.shape[1] < tilesize:
            #print('Padding %i' % (counter))
            subset_pad = np.zeros((tilesize, tilesize, subset.shape[-1]), dtype=profile['dtype'])
            x,y,z=subset.shape
            subset_pad[:x,:y,:z] = subset
            del subset
            subset = subset_pad.copy()
            del subset_pad

        #print('size: %i %i'% (subset.shape[0], subset.shape[1]))

        if counter<10:
            imsave('tmp/ortho/'+prefix+'000000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
        elif counter<100:
            imsave('tmp/ortho/'+prefix+'00000'+str(counter)+'.png', subset.astype(np.uint8), compression=0, check_contrast=False)
        elif counter<1000:
            imsave('tmp/ortho/'+prefix+'0000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
        elif counter<10000:
            imsave('tmp/ortho/'+prefix+'000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
        else:
            imsave('tmp/ortho/'+prefix+'00'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)

        del subset

#-----------------------------------
def mask_chunk(sample_filename, tilesize):
    """
    create mask from image chunk
    """
    threshold = 0.5

    if USE_RGB:
        image = seg_file2tensor_4band(sample_filename.replace('stdev', 'ortho'), sample_filename )/255
    else:
        image = seg_file2tensor_1band(sample_filename)/255

    if len(np.unique(image))>1:
        est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        K.clear_session()

        #confidence is computed as deviation from 0.5
        conf = 1-est_label
        conf[est_label<threshold] = est_label[est_label<threshold]
        conf = 1-conf
        #model_conf = np.sum(conf)/np.prod(conf.shape)
        #print('Overall model confidence = %f'%(model_conf))

        est_label[est_label<threshold] = 0
        est_label[est_label>threshold] = 1
        conf = (100*conf).astype('uint8')  #turn this into 8-bit for more efficient ram usage
        # conf[conf<50]=50
        # conf[conf>100]=100

    else:
        est_label = conf = None

    return est_label, conf
###==================================================================

# do_exp = True
tilesize = 1024 #
TARGET_SIZE = [tilesize, tilesize]
BATCH_SIZE = 6
NCLASSES = 1
DO_PARALLEL = True
USE_RGB = False #False = use dem only
OVERLAP = True
CALC_CONF = False #True

if USE_RGB:
    weights = 'orthoclip_2class_batch_6.h5'
else:
    weights = 'orthoclip_stdevonly_2class_batch_6.h5'

if DO_PARALLEL:
    from joblib import Parallel, delayed

#====================================================
print('.....................................')
print('.....................................')
print('.....................................')
print('.....................................')
print('.....................................')
print('.....................................')
print('................A.O.M................')
print('.....................................')
print('.....................................')
print('.....................................')
print('.....................................')
print('.....................................')
print('.....................................')


root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("DSM/DEM geotiff file","*.tif"),("all files","*.*")))
dem = root.filename
root.withdraw()
print("Working on %s" % (dem))

if USE_RGB:
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("ortho geotiff file","*.tif"),("all files","*.*")))
    rgb = root.filename
    root.withdraw()
    print("... and %s" % (rgb))

#====================================================
### step 1 : make chunks and save them to temporary png files
print('.....................................')
print('Making small image chunks ...')

with rasterio.open(dem) as src:
    profile = src.profile

width = profile['width'] #21920
height = profile['height'] #48990
prefix = id_generator()+'_' #'example_'

padwidth = width + (tilesize - width % tilesize)
padheight = height + (tilesize - height % tilesize)

try:
    os.mkdir('tmp')
    if USE_RGB:
        os.mkdir('tmp/ortho')
    os.mkdir('tmp/stdev')
except:
    pass

if DO_PARALLEL:
    if OVERLAP:
        ivec = range(0, padwidth, int(tilesize/2)) #25% overlap
        jvec = range(0, padheight, int(tilesize/2))
    else:
        ivec = range(0, padwidth, tilesize)
        jvec = range(0, padheight, tilesize)

    #w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(print)((counter,i,j)) for counter,(i,j) in enumerate(itertools.product(ivec,jvec)))
    if USE_RGB:
        w = Parallel(n_jobs=-1, verbose=0)(delayed(chunk_image)(i,j,tilesize, dem, USE_RGB, prefix, counter, rgb) for counter,(i,j) in enumerate(itertools.product(ivec,jvec)))
    else:
        w = Parallel(n_jobs=-1, verbose=0)(delayed(chunk_image)(i,j,tilesize, dem, USE_RGB, prefix, counter) for counter,(i,j) in enumerate(itertools.product(ivec,jvec)))

else:
    counter=0
    for i in tqdm(range(0, padwidth, tilesize)):
        for j in range(0, padheight, tilesize):

            if USE_RGB:
                chunk_image(i,j,tilesize, dem, USE_RGB, prefix, counter, rgb)
            else:
                chunk_image(i,j,tilesize, dem, USE_RGB, prefix, counter)

            counter+=1


#=======================================================
## step 2: make an instance of the model and load the weights

if USE_RGB:
    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 4), BATCH_SIZE, NCLASSES)
else:
    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 1), BATCH_SIZE, NCLASSES)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

model.load_weights(weights)

#=======================================================
## step 3: use the model for prediction
print('.....................................')
print('Using model for prediction on images ...')

sample_filenames = sorted(tf.io.gfile.glob('tmp/stdev'+os.sep+'*.png'))
if OVERLAP:
    print('Number of samples: %i (50 percent overlap)' % (len(sample_filenames)))
else:
    print('Number of samples: %i' % (len(sample_filenames)))

counter=0
if OVERLAP:
    out_mask = np.zeros((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.uint8)
    if CALC_CONF:
        out_conf = np.ones((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.uint8)
    n = np.zeros((padwidth+int(tilesize/2), padheight+int(tilesize/2)), dtype=np.uint8)
    for i in tqdm(range(0, width, int(tilesize/2))):
        for j in range(0, height, int(tilesize/2)):

            est_label, conf = mask_chunk(sample_filenames[counter], tilesize)
            #est_label[np.isnan(est_label)] = 0
            #est_label[np.isinf(est_label)] = 0
            if conf is not None:
                out_mask[i:i+tilesize,j:j+tilesize] += est_label.astype('uint8') #fill in that portion of big mask
                if CALC_CONF:
                    out_conf[i:i+tilesize,j:j+tilesize] += conf.astype('uint8') #fill out that portion of the big confidence raster
                n[i:i+tilesize,j:j+tilesize] += 1

            counter +=1
            if CALC_CONF:
                del conf
            del est_label

    #out_mask[out_mask==np.nan]=0
    out_mask = np.divide(out_mask, n+0.00001)# out_mask/n #divide out by number of times each cell was sampled
    out_mask = out_mask.astype('uint8')
    out_mask[out_mask>1] = 1

    out_mask = out_mask[:width,:height]
    out_mask = np.rot90(np.fliplr(out_mask))
    out_shape = out_mask.shape

    outfile = os.path.join(mkdtemp(), 'mask.dat')
    fp = np.memmap(outfile, dtype='uint8', mode='w+', shape=out_mask.shape)
    fp[:] = out_mask[:]
    fp.flush()
    del out_mask
    del fp
    gc.collect()

    if CALC_CONF:
        out_conf = out_conf/100
        out_conf = out_conf.astype('float32')
        out_conf = np.divide(out_conf, n+0.00001) #out_conf/n
        del n
        out_conf = out_conf.astype('float32')
        out_conf[out_conf==np.nan]=0
        out_conf[out_conf==np.inf]=0
        out_conf[out_conf>=1] = 0
        out_conf[out_conf<.5] = 0
        out_conf[out_conf>=.9] = 0
        out_conf[out_conf<.5] = 1
        out_conf = out_conf[:width,:height]
        out_conf = np.rot90(np.fliplr(out_conf))

else:

    out_mask = np.zeros((padwidth, padheight), dtype=np.uint8)
    if CALC_CONF:
        out_conf = np.ones((padwidth, padheight), dtype=np.uint8)
    for i in tqdm(range(0, width, tilesize)):
        for j in range(0, height, tilesize):

            est_label, conf = mask_chunk(sample_filenames[counter], tilesize)
            if conf is not None:
                out_mask[i:i+tilesize,j:j+tilesize] = est_label.astype('uint8') #fill in that portion of big mask
                if CALC_CONF:
                    out_conf[i:i+tilesize,j:j+tilesize] = conf.astype('uint8') #fill out that portion of the big confidence raster

            counter +=1
            if CALC_CONF:
                del conf
            del est_label

    out_mask = out_mask[:width,:height]
    out_mask = np.rot90(np.fliplr(out_mask))
    out_shape = out_mask.shape

    if CALC_CONF:
        out_conf = out_conf.astype('float32')
        out_conf = out_conf[:width,:height]
        out_conf = np.rot90(np.fliplr(out_conf))


# out_conf = out_conf.astype('float32')
# out_mask = out_mask.astype('uint8')


#=======================================================
# step 4, tify up
# delete tmp files
print('.....................................')
print('Removing temporary images ...')

if USE_RGB:
    for f in glob('tmp/ortho/*.*'):
        os.remove(f)
    os.rmdir('tmp/ortho')
for f in glob('tmp/stdev/*.*'):
    os.remove(f)
os.rmdir('tmp/stdev')
os.rmdir('tmp')
gc.collect()

#=======================================================
##step 5: write out masked dem and rgb ortho and mask to file

with rasterio.open(dem) as src: #rgb
    profile = src.profile

#=============================
##conf
if CALC_CONF:
    print('.....................................')
    print('Writing out mask confidence raster ...')

    # print(out_conf.shape)
    profile['dtype'] = 'float32'
    profile['nodata'] = 0.0
    profile['count'] = 1

    with rasterio.Env():
        with rasterio.open(dem.replace('_cog.tif','_automask_conf.tif'), 'w', **profile) as dst:
            dst.write(np.expand_dims(out_conf,0))
    del out_conf
    gc.collect()


if OVERLAP:
    out_mask = np.memmap(outfile, dtype='uint8', mode='r', shape=out_shape)

print(out_mask.shape)
#hatteras (13990, 21900)
#va (48990, 21920)

#=============================
## dem
print('.....................................')
print('Writing out masked DEM ...')

with rasterio.open(dem) as src: #rgb
    profile = src.profile
    bigtiff = src.read()

bigtiff = mask(bigtiff,profile,out_mask)

with rasterio.Env():
    with rasterio.open(dem.replace('_cog.tif','_automasked.tif'), 'w', **profile) as dst:
        dst.write(bigtiff)
del bigtiff
gc.collect()



#=============================
##mask
print('.....................................')
print('Writing out mask raster ...')

profile['nodata'] = 0.0
profile['dtype'] = 'uint8'
profile['count'] = 1

with rasterio.Env():
    with rasterio.open(dem.replace('_cog.tif','_automask.tif'), 'w', **profile) as dst:
        dst.write(np.expand_dims(out_mask,0))
    gc.collect()

#=============================
# mask orhto
if USE_RGB:
    with rasterio.open(rgb) as src: #rgb
        profile = src.profile
        bigtiff = src.read()

    bigtiff = mask(bigtiff,profile,out_mask)
    del out_mask

    #wrte maked geotiff to file
    with rasterio.Env():
        with rasterio.open(rgb.replace('_cog.tif','_automasked.tif'), 'w', **profile) as dst:
            dst.write(bigtiff)
    del bigtiff
    gc.collect()

else:
    from tkinter import messagebox
    root= Tk()
    MsgBox = messagebox.askquestion ('Mask ortho?','Would you like to use this mask to mask an ortho?',icon = 'warning')
    root.destroy()
    if MsgBox == 'yes':
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("ortho geotiff file","*.tif"),("all files","*.*")))
        rgb = root.filename
        root.withdraw()

        print('.....................................')
        print('Writing out masked RGB ortho ...')

        with rasterio.open(rgb) as src: #rgb
            profile = src.profile
            bigtiff = src.read()

        bigtiff = mask(bigtiff,profile,out_mask)
        del out_mask

        #wrte maked geotiff to file
        with rasterio.Env():
            with rasterio.open(rgb.replace('_cog.tif','_automasked.tif'), 'w', **profile) as dst:
                dst.write(bigtiff)
        del bigtiff
        gc.collect()

if 'outfile' in locals():
    os.remove(outfile)
