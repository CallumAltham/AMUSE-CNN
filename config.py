import argparse
import rasterio

# File for containing global variables
# Using import config in other files allows config.GLOBAL_VAR_NAME to be updated/accessed

TIF_PATH = None
TIF = None
IDXS_TIF = None
MODELS_DIRECTORY = None
LC_ANNOTATION_DIRECTORY = None
LU_ANNOTATION_DIRECTORY = None
SEGMENTATION_SHAPEFILE = None
NUMB_LC_CLASSES = None
NUMB_LU_CLASSES = None
WINDOW_SIZE = None
NUMB_SEGMENTS = None
ADAPTIVE_RUN = None
