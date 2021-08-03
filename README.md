## Auto DEM Masker (A.D.M)

Written by Daniel Buscombe (USGS-PCMSC and Marda Science LLC) and Andy Ritchie (USGS-PCMSC)

> version 1 - April 19, 2021

Initial version did not use overlap, not Otsu. Implemented a median filter with large radus (21 px) by default.

> version 2 - August 2nd, 2021

Optional overlap, Otsu threshold, median filter. Improved implementation. 


Input: DEM raster in COG (cloud-optimized geotiff) format. This DEM has messy noise you wish to automatically identify and clean.

Outputs:
1) a noise-free DEM
2) the mask used to create the noise-free DEM (1=no noise, 0=noise)
3) a confidence metric for each prediction in the mask (0.0 = uncertain, 0.5 = certain)


## How it works

The program uses a deep neural network trained to identify noise in DEMs, to identify and remove noise in DEMs

1. The user selects a DEM in geotiff format (including COG).

2. The DEM raster is indexed into 1024x1024 pixel windows

3. For each pixel in each window, the model estimates 'good' data as a probability score. Scores close to zero indicate 'bad' data (noise). Scores close to one indicate 'good' (noise-free)

4. Each band of the model output stack (one-hot encoded output probabilities) is median filtered to remove edge artifacts and spatially filter confidence estimates



## Install the conda env

We recommend using a dedicated conda environment to ensure you have the dependencies you need to run this program, isolated from your existing conda environments

```
conda env create --file install/auto_dem_masker.yml
conda activate adm
```

## Run the program

There are two options

```
python mask_dem.py
```

The program will ask you to select one or more DEM(s) in geotiff format


What it does:
1. Chops the dem into 1024x1024 px chunks
2. For each chunk, the model reads in the dem, and the model makes a prediction
3. The prediction is thresholded to make a mask, and a confidence is also computed
4. The masks and confidence chunks are compiled into mask and confidence rasters the same size as the input dem
5. The mask raster is used to mask the dem and the masked dem is written to file
6. The mask raster is written to file

It can take several minutes for this process to complete, and very large rasters require large amounts of RAM

Options (hard-coded into `mask_dem.py`)

```
USE_GPU = True
```
if `False`, CPU is used for inference


## Contents of this repository

* `download_test_images.py`: a script to download example data to use with the model if you have no data of your own
* `mask_dem.py`: the main script that does the DEM masking
* `model/orthoclip_demonly_3class_batch_6.json`: the model in human-readable format, consisting of a series of instructions in tensorflow/keras
* `model/orthoclip_demonly_3class_batch_6.h5`: the model trained weights
* `install/auto_dem_masker.yml`: a yml file that contains instructions for creating a conda environment that installs all the required dependencies
* segmentation zoo config files
* `chunk_dems.py`:the script to use to create model training data for use with segmentation zoo

## Updates

### 3/18/21: AR
* removed orthomosaic masking using file open dialog
* printed timer for individual processes
* changed temp file creation method for better cross-platform functionality

### 3/19/21: DB
* massively stripped down the code base. more memory efficient and faster
* single stage process - no intermediate file creation. Chunks with no data are ignored.
* OVERLAP = True (no alternative). 50% overlap is enforced
* memory mapping used for mask and confidence rasters
* overall program execution time in minutes printed to screen
* model now written out to json file and called rather than built from functions. Therefore fewer subfunctions
* USE_RGB no longer an option and the 4band model removed
* auto-batch! just put tifs in the folder and use `-m autobatch`
* removed joblib dependency (no parallel processing)
* removed yml file into install folder and json and h5 files into model folder. tidier
* command line flags either 'autobatch' for 'auto batch processing mode' that analyzes every .tif file in the present directory, or 'prompt' (default) tot select files one by one with a file browser dialog window
* now can select multiple tifs and batch process them
* otsu threshold

### 3/25/21: DB
* switched whole codebase from overlap to no overlap. Now seams are removed by smoothing, not oversampling
* switched from 2-class (good and bad/no data), now have 3 classes (good, bad, and no data). The goal is to isolate the 'good data' in a mask 
* removed autobatch mode therefore no need to specify mode at cl
* usr flag is now model type. None work except the default (dem only, m=1)
* some other minor improvements to mask_dem.py, added 'make_models.py', added segmentation zoo config files and 'chunk_dems.py' which is the script to use to create model training data for use with segmentation zoo
* adapted and test on 1, 3, and 4-band inputs/models, however currently all models except dem-only give bad predictions
* 3-class now so removed Otsu threshold

### 8/2/21: AR + DB
* back to overlap, but a different more efficent implementation
* have temporarily removed the confidence raster
* added switches to control threshold behaviour (o.5 or otsu)
* added switch for median filter (defailt - no median filter)
* tested with 25% overlap, no otsu, no median filter = worked well on Florence data
* 1=good data, 0=no/bad data
