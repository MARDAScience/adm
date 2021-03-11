## Auto Ortho Masker or A.O.M

Input: coincident orthomosaic and dem with messy noise

Output: a noise-free orthomosaic and dem, the mask used to make them, and a confidence metric for each prediction


### Install the conda env

```
conda env create --file auto_ortho_masker.yml
conda activate aom
```

### Run the program

```
python mask_dem.py
```

The program will ask you to select an Orthomosaic in geotiff format, then a DEM in the same format, defined over the same grid.

What it does:
1. chops the ortho into 1024x1024 px chunks and stores to png files in a tmp directory
2. chops the dem into 1024x1024 px chunks, computes the stdev raster using a 3x3 kernel, scales it empirically, and stores to png files in a tmp directory
3. For each chunk, the model reads in the ortho and dem, merges to make a 4-band raster, and the model makes a prediction
4. The prediction is thresholded to make a mask, and a confidence is also computed
5. The masks and confidence chunks are compiled into mask and confidence rasters the same size as the input orthos and dems
6. The mask raster is used to mask the dem and the masked dem is written to file
7. The mask raster is used to mask the ortho and the masked ortho is written to file
8. The mask raster is written to file
9. The confidence raster is written to file

Options (hard-coded into `mask_dem.py`)

```
USE_GPU = True
```
if `False`, CPU is used for inference

```
DO_PARALLEL = True
```
if `False`, images are written to file in serial (not recommended)

```
USE_RGB = False
```
if `True`, model will use DEM + RGB image for prediction. If `False` only DEM (recommended)

```
OVERLAP = True
```
If `True`, 50% overlap used for prediction. Slower, but generally more accurate

```
CALC_CONF = False
```

If `True`, a confidence raster will be computed
