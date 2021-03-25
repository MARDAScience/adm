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

import rasterio
from glob import glob
import numpy as np
from skimage.io import imsave, imread
from scipy.signal import convolve2d
from tkinter import filedialog, messagebox
from tkinter import *
import random, string, os

# =========================================================
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

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

def z_exp(z,n=0.5):
    zmin = np.min(z)
    zmax = np.max(z)
    zrange = zmax-zmin
    ivals = 255*((z-zmin)/(zrange))**n
    return ivals

######################## INPUTS
# do_exp = True
tilesize = 1024 #

#====================================================

keep_going = True
O = []; D = []; M = []
while keep_going is True:

    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("RGB ortho file","*.tif"),("all files","*.*")))
    rgb = root.filename
    root.withdraw()
    O.append(rgb)

    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("DSM/DEM ortho file","*.tif"),("all files","*.*")))
    dem = root.filename
    root.withdraw()
    D.append(dem)

    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("Mask ortho file","*.tif"),("all files","*.*")))
    mask = root.filename
    root.withdraw()
    M.append(mask)

    Tk().withdraw()
    keep_going = messagebox.askokcancel("","Would you like to chunk more data?")

try:
    os.mkdir('dems')
    os.mkdir('images')
    os.mkdir('masks')
    os.mkdir('stdev')
except:
    pass

for rgb,dem,mask in zip(O,D,M):
    print('Working on ortho %s |||| dem %s |||| and mask %s' % (rgb.split(os.sep)[-1], dem.split(os.sep)[-1], mask.split(os.sep)[-1]))

    with rasterio.open(mask) as src:
        profile = src.profile

    width = profile['width'] #21920
    height = profile['height'] #48990
    prefix = id_generator()+'_' #'example_'

    # i=0; j=13312

    counter=0
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):

            window = rasterio.windows.Window(i,j,tilesize, tilesize)

            with rasterio.open(dem) as src: #rgb
                profile = src.profile
                subset = src.read(window=window)

            orig = subset.squeeze()
            # set nodata to zero
            orig[orig==profile['nodata']] = 0
            # apply stdev filter to get stdev raster
            subset = std_convoluted(orig, 3)
            # scale with exponential
            subset = z_exp(subset,n=0.5)
            #print(np.max(subset))
            subset = np.squeeze(subset).T

            if np.max(orig)>0:
                # print((i,j))
                # print(np.max(subset))
                if counter<10:
                    imsave('stdev/'+prefix+'000000'+str(counter)+'_nir.png', subset.astype(np.uint8), compression=0, check_contrast=False)
                elif counter<100:
                    imsave('stdev/'+prefix+'00000'+str(counter)+'_nir.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
                else:
                    imsave('stdev/'+prefix+'0000'+str(counter)+'_nir.png', subset.astype(np.uint8),  compression=0, check_contrast=False)

                del subset

                with rasterio.open(rgb) as src: #rgb
                    #print(src.profile)
                    subset = src.read(window=window)

                subset = np.squeeze(subset).T
                if counter<10:
                    imsave('images/'+prefix+'000000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
                elif counter<100:
                    imsave('images/'+prefix+'00000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)
                else:
                    imsave('images/'+prefix+'0000'+str(counter)+'.png', subset.astype(np.uint8),  compression=0, check_contrast=False)

                del subset

                with rasterio.open(dem) as src: #dem
                    #print(src.profile)
                    d = src.read(window=window)
                    d[d==src.profile['nodata']] = 0

                d = np.squeeze(d).T
                if counter<10:
                    imsave('dems/'+prefix+'000000'+str(counter)+'_nir.png', d.astype(np.uint8),  compression=0, check_contrast=False)
                elif counter<100:
                    imsave('dems/'+prefix+'00000'+str(counter)+'_nir.png', d.astype(np.uint8),  compression=0, check_contrast=False)
                else:
                    imsave('dems/'+prefix+'0000'+str(counter)+'_nir.png', d.astype(np.uint8),  compression=0, check_contrast=False)

                #del subset


                with rasterio.open(mask) as src: #mask
                    #print(src.profile)
                    subset = src.read(window=window)
                    subset[subset==src.profile['nodata']] = 0
                subset = np.squeeze(subset).T

                subset[(subset==0) & (d!=0) ]=2 # third class = bad

                if counter<10:
                    imsave('masks/'+prefix+'000000'+str(counter)+'_mask.png', subset.astype(np.uint8)+1,  compression=0, check_contrast=False)
                elif counter<100:
                    imsave('masks/'+prefix+'00000'+str(counter)+'_mask.png', subset.astype(np.uint8)+1,  compression=0, check_contrast=False)
                else:
                    imsave('masks/'+prefix+'0000'+str(counter)+'_mask.png', subset.astype(np.uint8)+1,  compression=0, check_contrast=False)

                del subset


                counter +=1
