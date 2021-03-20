## Auto DEM Masker (A.D.M)

Written by Daniel Buscombe (USGS-PCMSC and Marda Science LLC) and Andy Ritchie (USGS-PCMSC)

Input: DEM raster in COG (cloud-optimized geotiff) format. This DEM has messy noise you wish to automatically identify and clean.

Outputs:
1) a noise-free DEM
2) the mask used to create the noise-free DEM (1=no noise, 0=noise)
3) a confidence metric for each prediction in the mask (0.5 = uncertain, 1.0 = certain)


## How it works

The program uses a deep neural network trained to identify noise in DEMs, to identify and remove noise in DEMs

1. The user selects a DEM in geotiff format (including COG).

2. The DEM raster is indexed into 1024x1024 pixel windows with 50% overlap. Overlap results in over-sampling the DEM with the model and averaging provides stability and confidence in outputs

3. For each pixel in each window, the model estimates 'good' data as a probability score. Scores close to zero indicate 'bad' data (noise). Scores close to one indicate 'good' (noise-free)

4. With 50% overlap, each pixel in the input DEM is sampled up to 4 times (twice in each direction), so the aggregated probabilities of 'good' data (model outputs from step 3 above) are averaged by the number of times each pixel was sampled by the model

5. The confidence of each model prediction is the distance from 0.5 (the uncertainty threshold). With overlap, confidences are aggregated and divided by the number of samples to provide an average confidence


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

The above is equivalent to

```
python mask_dem.py -m prompt
```

using the `-m` or mode flag for `prompt` mode. An alternative is

```
python mask_dem.py -m autobatch
```

which tells the program to analyze all the .tif files in the root directory.

Optionally, use the `-t` flag if you'd like to use the Otsu threshold instead of a hard threshold of 0.5 to separate good from bad data, e.g.

```
python mask_dem.py -t
```

or

```
python mask_dem.py -m autobatch -t
```


What it does:
1. Chops the dem into 1024x1024 px chunks, computes the stdev raster using a 3x3 kernel, scales it empirically
2. For each chunk, the model reads in the dem, and the model makes a prediction
3. The prediction is thresholded to make a mask, and a confidence is also computed
4. The masks and confidence chunks are compiled into mask and confidence rasters the same size as the input dem
5. The mask raster is used to mask the dem and the masked dem is written to file
6. The mask raster is written to file
7. The confidence raster is written to file

It can take several minutes for this process to complete, and very large rasters require large amounts of RAM (> 16GB or more)

Options (hard-coded into `mask_dem.py`)

```
USE_GPU = True
```
if `False`, CPU is used for inference

```
CALC_CONF = False
```

If `True`, a confidence raster will be computed

## Contents of this repository

* `download_test_images.py`: a script to download example data to use with the model if you have no data of your own
* `mask_dem.py`: the main script that does the DEM masking
* `model/orthoclip_stdevonly_2class_batch_6.json`: the model in human-readable format, consisting of a series of instructions in tensorflow/keras
* `model/orthoclip_stdevonly_2class_batch_6.h5`: the model trained weights
* `install/auto_dem_masker.yml`: a yml file that contains instructions for creating a conda environment that installs all the required dependencies


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
* tested on 15 DEMs each < 257 MB. Good outputs for:
  * 20181007_Hattaras_Inlet_to_Ocracoke_Inlet_1m_DSM_adjExt_cog.tif
  * 20181006_VA_to_Oregon_Inlet_1m_DSM_adjExt_cog.tif
  * clip_20181006_Ophelia_Inlet_to_Beaufort_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181007_Hatteras_Inlet_to_Ocracoke_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181007_Oregon_Inlet_to_Hatteras_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181008_Hatteras_Inlet_to_Ocracoke_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181008_VA_to_Oregon_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181007_Ophelia_Inlet_to_Beaufort_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181007_Ocracoke_Inlet_to_Ophelia_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181007_Bogue_Inlet_to_New_River_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181007_Beaufort_Inlet_to_Bogue_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181006_VA_to_Oregon_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181006_Ophelia_Inlet_to_Beaufort_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181006_Ocracoke_Inlet_to_Ophelia_Inlet_1m_UTM18N_NAVD88_cog.tif
  * clip_20181006_Hatteras_Inlet_to_Ocracoke_Inlet_1m_UTM18N_NAVD88_cog.tif
