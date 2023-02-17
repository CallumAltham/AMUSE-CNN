from numpy.random import choice
import housekeeping # Used to reduce verbose output and memory usage in testing, can be removed if necessary. Source: https://github.com/CallumAltham/Tensorflow-NVIDIA-Fix
from ortools.algorithms import pywrapknapsack_solver
from shapely.geometry import box, shape
import argparse
import rasterio
from rasterio.io import MemoryFile
from sklearn.utils import shuffle
from sklearn.svm import SVC
from scipy.stats import describe
from rasterio.mask import mask
import numpy as np
import fiona
import tensorflow as tf
import glob
import json
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from window_positioner import segment_centroids
import os
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from time import time
import geopandas as gpd
from tensorflow.keras.utils import to_categorical
from multi_scale_ocnn import train_multiscale_model, MultiScaleModel, BenchmarkModel, train_adaptive_multiscale_model, train_ocnn_model, train_dual_multiscale_model
import config
import random

class RGBContainer:
    """Class used to contain and handle information about the image data in a memory-efficent manner.
    The SegmentContainer.prob_container is a LUProbabilityContainer object (for details about this class see its
    docstring.
    Rather than storing memory-intensive floats, the probabilities are represented using np.uint16 values (so if
    probability is 0, 0.5 or 1 it is stored as 0, 32767, or 65535 respectively within the array)
    """

    def __init__(self, runtype=None):
        # Create optical channels containing TIF data and save to disk
        self.optical_channels = np.rollaxis(config.TIF.read(), 0, 3).reshape(-1, 3)
        if runtype == None:
            with open(config.MODELS_DIRECTORY + '/rgb_c.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive":
            with open(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/rgb_c_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive-full":
            with open(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/rgb_c_{str(config.WINDOW_SIZE)}.pkl', 'wb') as f:
                pickle.dump(self, f)

    def get_one_rgb(self, pixel_idx):
        return (self.optical_channels[pixel_idx] / 255).astype(np.float32)

    def get_many_rgb(self, idxs_array):
        return (self.optical_channels[idxs_array] / 255).astype(np.float32)


class SegmentContainer:
    """Class used to contain and handle information about the data segments/superpixels.
    The SegmentContainer.prob_container is a LUProbabilityContainer object (for details about this class see its
    docstring."""

    def __init__(self, runtype=None):
        print("Opening segmentation shapefile...")
        # Load segmentation shapefile into memory using GeoPandas
        segments_gdf = gpd.read_file(config.SEGMENTATION_SHAPEFILE)
        # Create segment maop by creating an array of 0s in the shape of the TIF file
        self.segment_map = np.zeros(config.TIF.shape[0] * config.TIF.shape[1], dtype=np.uint32)
        print("Computing segment map...")
        # Loop through index and shape of geometry values of segmentation shapefile
        for i, one_shape in tqdm(enumerate(segments_gdf.geometry.values)):
            # Get all pixels found within the shape as a list of indices
            segment_pixel_idxs = get_shape_pixel_idxs(one_shape)
            # Set segment pixel indexes as element of segment map containing numpy array with size of index * length of pixel index list
            self.segment_map[segment_pixel_idxs] = np.array([i] * len(segment_pixel_idxs), dtype=np.uint32)
        print("Initiating segments...")
        # Create LU prob container with length of segmentation shapefile 
        self.prob_container = LUProbabilityContainer(len(segments_gdf))
        print("Computing segment window locations...")

        # Get centroids based upon each shape in segmentation shapefile
        centroids = [segment_centroids(one_shape, config.WINDOW_SIZE, config.TIF.transform[0]) for one_shape in
                     tqdm(segments_gdf.geometry.values)]
        self.c_idx_to_s_idx = {c: seg_idx for seg_idx, c_list in enumerate([[a + i for i in range(len(b))] for a, b in
                                                                            zip(np.cumsum(
                                                                                [0] + [len(s) for s in centroids]),
                                                                                [s for s in centroids])]) for c in
                               c_list}
        
        # Set segment tiles as array of segment in centroids list
        self.segment_tiles = np.array([c for seg in centroids for c in seg])
        # Save segment container in pickle format
        if runtype == None:
            with open(config.MODELS_DIRECTORY + '/seg_c.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive":
            with open(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/seg_c_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive-full":
            with open(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/seg_c_{str(config.WINDOW_SIZE)}.pkl', 'wb') as f:
                pickle.dump(self, f)

    def get_seg_idx(self, pixel_idx):
        return self.segment_map[pixel_idx]

    def get_many_lu_p_from_seg_idxs(self, seg_idxs_array):
        return self.prob_container.get_many_p(seg_idxs_array)

    def get_many_lu_p_from_pixel_idxs(self, pixel_idxs_array):
        return self.prob_container.get_many_p(self.segment_map[pixel_idxs_array])

    def get_tile_pixel_idxs(self, centroid_idx):
        j, i = ~config.TIF.transform * self.segment_tiles[centroid_idx]
        i, j = int(i), int(j)
        i_list = [i_shift + i for i_shift in range(-int(config.WINDOW_SIZE / 2), int(config.WINDOW_SIZE / 2))]
        i_list = [a if 0 <= a < config.TIF.shape[0] else -1 for a in i_list]
        j_list = [j_shift + j for j_shift in range(-int(config.WINDOW_SIZE / 2), int(config.WINDOW_SIZE / 2))]
        j_list = [a if 0 <= a < config.TIF.shape[1] else -1 for a in j_list]
        return np.array(
            [config.TIF.shape[1] * each_i + each_j if each_i != -1 and each_j != -1 else -1 for each_i in i_list for
             each_j in j_list], dtype=np.int32)


class LCProbabilityContainer:
    """Stores LC probabilities in a memory-efficient manner.
    LCProbabilityContainer.probabilities has shape (image_height*image_width, config.NUMB_LC_CLASSES) such that
    LCProbabilityContainer.probabilities[i] contains the i-th pixel's LC probabilities for each class.
    Rather than storing memory-intensive floats, the probabilities are represented using np.uint16 values (so if
    probability is 0, 0.5 or 1 it is stored as 0, 32767, or 65535 respectively within the array)
    """

    def __init__(self, sc, lc_am_object, lu_am_object, runtype=None):
        self.d_type = np.uint16

        # Create multi dimensional array with shape rows = tif.shape[0] * tif.shape[1] and row items = number of LC classes
        self.probabilities = np.zeros((config.TIF.shape[0] * config.TIF.shape[1], config.NUMB_LC_CLASSES),
                                      dtype=self.d_type)
        
        if runtype == None:
            # If relevant pixels exist, open files and load in
            if os.path.isfile(config.MODELS_DIRECTORY + '/relevant_pixels.pkl'):
                with open(config.MODELS_DIRECTORY + '/relevant_pixels.pkl', 'rb') as f:
                    self.relevant_pixels = pickle.load(f)
            else:
                # If not exist: 
                # Get relevant pixels based on segment container land cover 
                # annotation manager and land use annotation manager
                self.relevant_pixels = self.get_relevant_pixels(sc, lc_am_object, lu_am_object)
                # Save relevant pixels to disk
                with open(config.MODELS_DIRECTORY + '/relevant_pixels.pkl', 'wb') as f:
                    pickle.dump(self.relevant_pixels, f)

        elif runtype == "adaptive":
            # If relevant pixels exist, open files and load in
            if os.path.isfile(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/relevant_pixels_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl'):
                with open(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/relevant_pixels_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'rb') as f:
                    self.relevant_pixels = pickle.load(f)
            else:
                # If not exist: 
                # Get relevant pixels based on segment container land cover 
                # annotation manager and land use annotation manager
                self.relevant_pixels = self.get_relevant_pixels(sc, lc_am_object, lu_am_object)
                # Save relevant pixels to disk
                with open(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/relevant_pixels_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'wb') as f:
                    pickle.dump(self.relevant_pixels, f)

        elif runtype == "adaptive-full":
            # If relevant pixels exist, open files and load in
            if os.path.isfile(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/relevant_pixels_{str(config.WINDOW_SIZE)}.pkl'):
                with open(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/relevant_pixels_{str(config.WINDOW_SIZE)}.pkl', 'rb') as f:
                    self.relevant_pixels = pickle.load(f)
            else:
                # If not exist: 
                # Get relevant pixels based on segment container land cover 
                # annotation manager and land use annotation manager
                self.relevant_pixels = self.get_relevant_pixels(sc, lc_am_object, lu_am_object)
                # Save relevant pixels to disk
                with open(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/relevant_pixels_{str(config.WINDOW_SIZE)}.pkl', 'wb') as f:
                    pickle.dump(self.relevant_pixels, f)

    @staticmethod
    def get_relevant_pixels(sc, lc_am_object, lu_am_object):
        print("Obtaining pixels...")
        # Create set for pixel indexes
        relevant_pixel_idxs = set()

        # Loop through segment centroids
        for seg_centroid_idx in tqdm(range(sc.segment_tiles.shape[0])):
            # Loop through pixels in pixel index list of segment centroid
            for one_pixel in sc.get_tile_pixel_idxs(seg_centroid_idx):
                # Update relevant pixel indexes with pixel
                relevant_pixel_idxs.update({one_pixel})
        # Loop through polygon lists in land class annotation manager
        for poly_list in tqdm(lc_am_object.class_polygons.values()):
            # Loop through polygon in polygon list
            for one_poly in poly_list:
                # Loop through pixel in polygon pixel indexes
                for one_pixel in one_poly.pixel_idxs:
                    # Update relevant pixel indexes to contain pixels
                    relevant_pixel_idxs.update({one_pixel})
        # Loop through polygon lists in land use annotation manager
        for poly_list in tqdm(lu_am_object.class_polygons.values()):
            # Loop through polygon in polygon list
            for one_poly in poly_list:
                # Loop through polygon tiles
                for tile_idx in range(one_poly.tiles.shape[0]):
                    # Loop through pixel indexes in polygon tiles
                    for one_pixel in one_poly.get_one_tile_pixel_idxs(tile_idx):
                        # Update relevant pixels to contain pixels
                        relevant_pixel_idxs.update({one_pixel})
        # If -1 is found in relevant pixel indexes, remove
        if -1 in relevant_pixel_idxs:
            relevant_pixel_idxs.remove(-1)
        # Return pixel list 
        return np.array(list(relevant_pixel_idxs), dtype=np.uint32)

    def get_p(self, pixel_idx):
        if pixel_idx != -1:
            return self.probabilities[pixel_idx] / np.iinfo(self.d_type).max
        else:
            return np.zeros(config.NUMB_LC_CLASSES, dtype=np.float32)

    def get_many_p(self, idxs_array):
        return self.probabilities[idxs_array] / np.iinfo(self.d_type).max

    def get_tile_p(self, tile_pixel_idxs):
        output = np.zeros((config.WINDOW_SIZE ** 2, config.NUMB_LC_CLASSES), dtype=np.float32)
        output[tile_pixel_idxs != -1] = self.get_many_p(tile_pixel_idxs[tile_pixel_idxs != -1])
        return output.reshape((config.WINDOW_SIZE, config.WINDOW_SIZE, config.NUMB_LC_CLASSES))

    def set_many_p(self, idxs_array, p):
        self.probabilities[idxs_array] = np.round(p * np.iinfo(np.uint16).max).astype(np.uint16)

    def update_pixels(self, mlp_object, rgb_c, seg_c=None, all_pixels=False):
        print("Getting MLP output for next step...")
        if not all_pixels:
            numb_pixels = len(self.relevant_pixels)
        else:
            numb_pixels = self.probabilities.shape[0]
        if seg_c is None:
            for i in tqdm(range((numb_pixels - 1) // config.TIF.shape[0] + 1)):
                if not all_pixels:
                    model_output = mlp_object.predict(
                        rgb_c.get_many_rgb(self.relevant_pixels[i * config.TIF.shape[0]:(i + 1) * config.TIF.shape[0]]),
                        as_probabilities=True)
                    self.set_many_p(self.relevant_pixels[i * config.TIF.shape[0]:(i + 1) * config.TIF.shape[0]],
                                    model_output)
                else:
                    model_output = mlp_object.predict(
                        rgb_c.get_many_rgb(np.arange(i * config.TIF.shape[0], (i + 1) * config.TIF.shape[0])),
                        as_probabilities=True)
                    self.set_many_p(np.arange(i * config.TIF.shape[0], (i + 1) * config.TIF.shape[0]), model_output)

        else:
            for i in tqdm(range((numb_pixels - 1) // config.TIF.shape[0] + 1)):
                if not all_pixels:
                    model_output = mlp_object.predict(
                        np.hstack((rgb_c.get_many_rgb(
                            self.relevant_pixels[i * config.TIF.shape[0]:(i + 1) * config.TIF.shape[0]]),
                                   seg_c.get_many_lu_p_from_pixel_idxs(
                                       self.relevant_pixels[i * config.TIF.shape[0]:(i + 1) * config.TIF.shape[0]]))),
                        as_probabilities=True)
                    self.set_many_p(self.relevant_pixels[i * config.TIF.shape[0]:(i + 1) * config.TIF.shape[0]],
                                    model_output)
                else:
                    model_output = mlp_object.predict(
                        np.hstack(
                            (rgb_c.get_many_rgb(np.arange(i * config.TIF.shape[0], (i + 1) * config.TIF.shape[0])),
                             seg_c.get_many_lu_p_from_pixel_idxs(
                                 np.arange(i * config.TIF.shape[0], (i + 1) * config.TIF.shape[0])))),
                        as_probabilities=True)
                    self.set_many_p(np.arange(i * config.TIF.shape[0], (i + 1) * config.TIF.shape[0]), model_output)


class LUProbabilityContainer:
    """Stores LU probabilities in a memory-efficient manner.
    LUProbabilityContainer.probabilities has shape (numb_segments, config.NUMB_LU_CLASSES) such that
    LUProbabilityContainer.probabilities[i] contains the i-th segments LU probabilities for each class.
    Rather than storing memory-intensive floats, the probabilities are represented using np.uint16 values (so if
    probability is 0, 0.5 or 1 it is stored as 0, 32767, or 65535 respectively within the array)
    """

    def __init__(self, numb_segments):
        self.d_type = np.uint16
        # Set probabilities as multi-dimensional array with shape rows = numb_segments, row items = numb_lu_classes 
        self.probabilities = np.zeros((numb_segments, config.NUMB_LU_CLASSES), dtype=self.d_type)

    def get_p(self, segment_idx):
        return (self.probabilities[segment_idx] / np.iinfo(self.d_type).max).astype(np.float32)

    def get_many_p(self, idxs_array):
        return (self.probabilities[idxs_array] / np.iinfo(self.d_type).max).astype(np.float32)

    def set_many_p(self, idxs_array, p):
        self.probabilities[idxs_array] = np.round(p * np.iinfo(np.uint16).max).astype(np.uint16)

    def update_segments(self, ocnn_model, lc_p, seg_c, rgb_c=None):
        print(f"Getting output for next step...")
        # Set chunk size
        chunk_size = 1024
        # Set numb segments based on prob container number
        numb_segments = seg_c.prob_container.probabilities.shape[0]

        # Loop through range of numb_segments divided by chunk_size + 1
        for i in tqdm(range(numb_segments // chunk_size + 1)):
            # Set number of tiles as chunk size if index isn't same as segments
            # divided by chunk size, else set it as modulus of numb segments 
            # and chunk size
            numb_tiles = chunk_size if i != numb_segments // chunk_size else numb_segments % chunk_size
            
            # Get centroid indexes
            centroid_idxs = [c_idx for c_idx in seg_c.c_idx_to_s_idx if
                             i * chunk_size <= seg_c.c_idx_to_s_idx[c_idx] < i * chunk_size + numb_tiles]
            # If no RGB container
            if rgb_c is None:
                # Output is set as OCNN prediction on tiles
                model_output = ocnn_model.predict(
                    np.array(
                        [lc_p.get_tile_p(seg_c.get_tile_pixel_idxs(centroid_idx)) for centroid_idx in centroid_idxs]))
            else:
                # Create input list
                model_input = []

                # Loop through centroid indexes
                for centroid_idx in centroid_idxs:
                    # Get pixel indexes
                    pixel_idxs = seg_c.get_tile_pixel_idxs(centroid_idx)
                    # Get segment channel 
                    seg_channel = seg_c.segment_map[pixel_idxs].reshape(
                        config.WINDOW_SIZE, config.WINDOW_SIZE, 1)
                    # Simplify segment dictionary
                    simplify_seg_dict = {s: i for i, s in enumerate(set(seg_channel.flatten()))}
                    for s in simplify_seg_dict:
                        seg_channel[seg_channel == s] = simplify_seg_dict[s]
                    # Append model input to list
                    model_input.append(
                        np.concatenate(
                            (rgb_c.get_many_rgb(pixel_idxs).reshape(config.WINDOW_SIZE, config.WINDOW_SIZE, 3),
                             seg_channel - seg_channel.min()),
                            axis=-1))
                # Stack input
                model_input = np.stack(model_input)
                # Get output by predicting against input 
                model_output = ocnn_model.predict(model_input[..., :3])
            self.set_many_p(np.arange(i * chunk_size, i * chunk_size + numb_tiles, dtype=np.uint32),
                            self.aggregate_multi_window_predictions(seg_c.c_idx_to_s_idx, model_output, centroid_idxs))

    @staticmethod
    def aggregate_multi_window_predictions(c_to_s_idx_dict, model_output, centroid_idxs):
        segments = {c_to_s_idx_dict[c_idx] for c_idx in centroid_idxs}
        array = np.zeros((len(segments), model_output.shape[1]))
        for i, seg_idx in enumerate(segments):
            array[i] = model_output[
                [idx for idx, c_idx in enumerate(centroid_idxs) if c_to_s_idx_dict[c_idx] == seg_idx]].mean(
                axis=0)
        return array


class LandUsePolygon:
    """Class which stores attributes pertaining to a LU polygon. Instantiated with the geometry of an annotated polygon,
    it generates sample tiles within the annotated region i.e. one annotated polygon can be responsible for many data
    samples for that class"""

    def __init__(self, geometry, tile_density):
        if tile_density == 'very low':
            self.tiles = np.array(get_centroids(geometry, 140, 0.5))
        elif tile_density == 'low':
            self.tiles = np.array(get_centroids(geometry, 90, 0.5))
        elif tile_density == 'medium':
            self.tiles = np.array(get_centroids(geometry, 40, 0.5))
        else:
            self.tiles = np.array(get_centroids(geometry, 20, 0.5))

    def get_one_tile_pixel_idxs(self, tile_idx):
        j, i = ~config.TIF.transform * self.tiles[tile_idx]
        i, j = int(i), int(j)
        i_list = [i_shift + i for i_shift in range(-int(config.WINDOW_SIZE / 2), int(config.WINDOW_SIZE / 2))]
        i_list = [a if 0 <= a < config.TIF.shape[0] else -1 for a in i_list]
        j_list = [j_shift + j for j_shift in range(-int(config.WINDOW_SIZE / 2), int(config.WINDOW_SIZE / 2))]
        j_list = [a if 0 <= a < config.TIF.shape[1] else -1 for a in j_list]
        return np.array(
            [config.TIF.shape[1] * each_i + each_j if each_i != -1 and each_j != -1 else -1 for each_i in i_list for
             each_j
             in j_list], dtype=np.int32)


class LandUseAnnotationManager:
    """Class which stores LU annotations and manages functions relating to them"""

    def __init__(self, k_fold_i=0, runtype=None):
        print("Loading Land Use annotations...")
        
        # Load all land use annotations and file paths for .shp files in alphabetical order

        file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
        self.class_polygons = {}
        self.class_training_idxs = {}
        
        # Loop through list of file paths in alphabetical order

        for i, path in enumerate(file_paths):

            # Open shape file
            with fiona.open(path) as shapefile:
                # Get all shapes contained within shape file
                shapes = [shape(feature['geometry']) for feature in shapefile]

                # Get list containing indexes of feature in shapefile if feature properties matches fold value i
                is_training = [idx for idx, feature in enumerate(shapefile) if
                               int(feature['properties']['k_fold_i']) != k_fold_i]
                
                # Get labelled area as sum of area of all shapes in shapes list
                labelled_area = sum([one_shape.area for one_shape in shapes])
                polygons = []

                # Loop through all shapes in shapes list using TQDM 
                for one_shape in tqdm(shapes):
                    # Generate land use tiles based upon tile density and store in polygons list
                    if labelled_area < 300000:
                        polygons.append(LandUsePolygon(one_shape, tile_density='high'))
                    elif labelled_area < 1000000:
                        polygons.append(LandUsePolygon(one_shape, tile_density='medium'))
                    elif labelled_area < 3000000:
                        polygons.append(LandUsePolygon(one_shape, tile_density='low'))
                    else:
                        polygons.append(LandUsePolygon(one_shape, tile_density='very low'))

            # Update class polygons dictionary with index of shapefile and polygons list            
            self.class_polygons.update({i: polygons})
            # Update class training dictionary with index of shapefile and list of features for training
            self.class_training_idxs.update({i: is_training})
        # Save land use annotation manager class to disk in pickle format    
        if runtype == None:
            with open(config.MODELS_DIRECTORY + '/lu_am.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive":
            with open(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/lu_am_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive-full":
            with open(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/lu_am_{str(config.WINDOW_SIZE)}.pkl', 'wb') as f:
                pickle.dump(self, f)
        

    # Get input and output of training based on land class prob container and
    # rgb container
    def get_training_ip_op(self, lc_prob_c, rgb_c=None):
        # Create input and output lists
        model_input, model_output = [], []

        # Loop through all class training indexes using TQDM
        for class_idx in tqdm(self.class_training_idxs):
            # Loop through polygons in training indexes
            for poly_idx in self.class_training_idxs[class_idx]:
                # Loop through tiles in polygon
                for tile_idx in range(self.class_polygons[class_idx][poly_idx].tiles.shape[0]):
                    # Get pixels for current pixel tile
                    pixel_idxs = self.class_polygons[class_idx][poly_idx].get_one_tile_pixel_idxs(tile_idx)
                    # If no RGB container, append pixels of tile
                    if rgb_c is None:
                        model_input.append(lc_prob_c.get_tile_p(pixel_idxs))
                    else:

                        # If RGB container exists

                        # model_input.append(
                        #     np.concatenate((rgb_c.get_many_rgb(pixel_idxs).reshape(config.WINDOW_SIZE, config.WINDOW_SIZE, 3),
                        #                     lc_prob_c.get_tile_p(pixel_idxs)), axis=-1))
                        
                        # Set segment channel as segment map element
                        seg_channel = segment_container.segment_map[pixel_idxs].reshape(config.WINDOW_SIZE,
                                                                                        config.WINDOW_SIZE, 1)

                        # Simplify segmentation dictionary by flattening seg_channel
                        simplify_seg_dict = {s: i for i, s in enumerate(set(seg_channel.flatten()))}
                        for s in simplify_seg_dict:
                            seg_channel[seg_channel == s] = simplify_seg_dict[s]

                        # Append to model input containing rgb container pixels
                        model_input.append(
                            np.concatenate(
                                (rgb_c.get_many_rgb(pixel_idxs).reshape(config.WINDOW_SIZE, config.WINDOW_SIZE, 3),
                                 seg_channel - seg_channel.min()),
                                axis=-1))
            # Extend output to contain class index
            model_output.extend([class_idx] * (len(model_input) - len(model_output)))
        return np.array(model_input, dtype=np.float32), np.array(model_output, np.uint8)

    # Basically identical to function above but for validation data over training
    def get_validation_ip_op(self, lc_prob_c, rgb_c=None):
        model_input, model_output = [], []
        for class_idx in tqdm(self.class_training_idxs):
            for poly_idx in range(len(self.class_polygons[class_idx])):
                if poly_idx not in self.class_training_idxs[class_idx]:
                    for tile_idx in range(self.class_polygons[class_idx][poly_idx].tiles.shape[0]):
                        pixel_idxs = self.class_polygons[class_idx][poly_idx].get_one_tile_pixel_idxs(tile_idx)
                        if rgb_c is None:
                            model_input.append(lc_prob_c.get_tile_p(pixel_idxs))
                        else:
                            # model_input.append(
                            #     np.concatenate((rgb_c.get_many_rgb(pixel_idxs).reshape(config.WINDOW_SIZE, config.WINDOW_SIZE, 3),
                            #                     lc_prob_c.get_tile_p(pixel_idxs)), axis=-1))
                            seg_channel = segment_container.segment_map[pixel_idxs].reshape(config.WINDOW_SIZE,
                                                                                            config.WINDOW_SIZE, 1)
                            simplify_seg_dict = {s: i for i, s in enumerate(set(seg_channel.flatten()))}
                            for s in simplify_seg_dict:
                                seg_channel[seg_channel == s] = simplify_seg_dict[s]
                            model_input.append(
                                np.concatenate(
                                    (rgb_c.get_many_rgb(pixel_idxs).reshape(config.WINDOW_SIZE, config.WINDOW_SIZE, 3),
                                     seg_channel - seg_channel.min()),
                                    axis=-1))
            model_output.extend([class_idx] * (len(model_input) - len(model_output)))
        return np.array(model_input, dtype=np.float32), np.array(model_output, np.uint8)


class LandCoverPolygon:
    """Class which stores attributes pertaining to a LC polygon"""

    def __init__(self, geometry):
        self.lc_class_value = None
        self.geometry = geometry
        self.pixel_idxs = get_shape_pixel_idxs(geometry)


class LandCoverAnnotationManager:
    """Class which stores LC annotations and manages functions relating to them"""

    def __init__(self, runtype=None):
        print("Loading Land Cover annotations...")

        # Loads all annotation .shp files from LC directory in alphabetical order into list

        file_paths = sorted([path for path in glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
        self.class_pixels = {}
        self.class_polygons = {}
        self.class_training_idxs = {}

        # with open('dlt_lc_lagos.json', 'r') as f:
        #     temp = json.load(f)
        #     self.class_training_idxs = {int(i): temp[i] for i in temp}

        # Loop through all .shp file paths

        for i, path in enumerate(file_paths):
            # Open shapefile
            with fiona.open(path) as shapefile:
                pixels = []
                polygons = []
                for feature in shapefile:
                    # Get shape of each feature in shapefile
                    polygon = shape(feature['geometry'])
                    # Find the pixels within the shape and store them as a list of indices
                    pixels.append(get_shape_pixel_idxs(polygon))
                    # Store polygon containing class value, geometry and pixel idxs
                    polygons.append(LandCoverPolygon(polygon))
            # Update class polygons dict to contain index of file and it's polygons
            self.class_polygons.update({i: polygons})
            # Update class pixels dict to contain all pixels from each list in shape file
            self.class_pixels.update({i: [item for sublist in pixels for item in sublist]})
            # Update class training idxs with index of file, and individual polygons 
            self.class_training_idxs.update(
                {i: get_training_polygon_idxs([one_polygon.geometry.area for one_polygon in polygons])})
        # Store class as pickle file with all annotations and pixels.        
        if runtype == None:
            with open(config.MODELS_DIRECTORY + '/lc_am.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive":
            with open(config.MODELS_DIRECTORY + f'/temp_data/{str(config.WINDOW_SIZE)}/{str(config.ADAPTIVE_RUN)}/lc_am_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif runtype == "adaptive-full":
            with open(config.MODELS_DIRECTORY + f'/new_data/{str(config.WINDOW_SIZE)}/lc_am_{str(config.WINDOW_SIZE)}.pkl', 'wb') as f:
                pickle.dump(self, f)

    def get_training_ip_op(self, rgb_c, seg_c, first_iteration=False):
        model_input, model_output = [], []
        for class_idx in tqdm(self.class_training_idxs):
            for poly_idx in self.class_training_idxs[class_idx]:
                pixel_idxs = self.class_polygons[class_idx][poly_idx].pixel_idxs
                if first_iteration:
                    model_input.append(rgb_c.get_many_rgb(pixel_idxs))
                else:
                    model_input.append(np.hstack((rgb_c.get_many_rgb(pixel_idxs),
                                                  seg_c.get_many_lu_p_from_pixel_idxs(pixel_idxs))))
            model_output.extend([class_idx] * (sum([a.shape[0] for a in model_input]) - len(model_output)))
        return np.concatenate(model_input), np.array(model_output, np.uint8)

    def get_validation_ip_op(self, rgb_c, seg_c, first_iteration=False):
        model_input, model_output = [], []
        for class_idx in tqdm(self.class_training_idxs):
            for poly_idx in range(len(self.class_polygons[class_idx])):
                if poly_idx not in self.class_training_idxs[class_idx]:
                    pixel_idxs = self.class_polygons[class_idx][poly_idx].pixel_idxs
                    if first_iteration:
                        model_input.append(rgb_c.get_many_rgb(pixel_idxs))
                    else:
                        model_input.append(np.hstack((rgb_c.get_many_rgb(pixel_idxs),
                                                      seg_c.get_many_lu_p_from_pixel_idxs(pixel_idxs))))
            model_output.extend([class_idx] * (sum([a.shape[0] for a in model_input]) - len(model_output)))
        return np.concatenate(model_input), np.array(model_output, np.uint8)


def make_idxs_tif():
    """Make a rasterio tif object but it has only one channel and it contains the unique pixel indices"""
    profile = config.TIF.profile
    profile.update({'dtype': rasterio.int32, 'count': 1, 'nodata': -1})
    with MemoryFile() as memory_file:
        tif = memory_file.open(**profile)
        tif.write(np.arange(config.TIF.shape[0] * config.TIF.shape[1]).astype(rasterio.int32).reshape(config.TIF.shape),
                  1)
    return tif


def get_shape_pixel_idxs(geometry):
    """Given a shapely shape, find the pixels within it and return as a list of indices (where each pixel in the domain
    is given a unique index corresponding to np.arange(image_height*image_width).reshape(image_height, image_width)"""
    if not isinstance(geometry, list):
        geometry = [geometry]
    elements = mask(config.IDXS_TIF, geometry, crop=True, nodata=-1, pad=True)[0]
    elements = elements.reshape(elements.shape[0], -1).T
    elements = np.delete(elements, np.where(elements == -1)[0], axis=0)
    return list(elements.flatten())


def get_centroids(geometry, shift=35, overlap_threshold=0.4):
    """Function used to take a annotated polygon and sample locations in it which will be used as training/validation
    data samples.
    Works by considering an array of locations on a grid spaced shift pixels vertically/horizontally from one another.
    If a square window centred on one of these locations overlaps the annotated polygon at least the overlap_threshold,
    then window of the data is considered to belong to the same class label as the annotated polygon so can be used as
    a training/validation data sample."""
    west, south, east, north = geometry.bounds  # in world coordinates
    x_min, y_min = ~config.TIF.transform * (west, south)  # in array coordinates
    x_max, y_max = ~config.TIF.transform * (east, north)
    # Note y_min > y_max because of image coordinates
    centroids = []
    for centroid_x in range(int(x_min), int(x_max), shift):
        for centroid_y in range(int(y_max), int(y_min), shift):
            box_west, box_north = config.TIF.transform * (centroid_x - 24, centroid_y - 24)
            box_east, box_south = config.TIF.transform * (centroid_x + 24, centroid_y + 24)
            # This tile is just arbitrary for checking an overlap area threshold
            tile = box(box_west, box_south, box_east, box_north)
            if intersection(tile, geometry).area / tile.area > overlap_threshold:
                centroids.append(((box_west + box_east) / 2, (box_north + box_south) / 2))
    if len(centroids) > 0:
        return centroids
    else:
        return [geometry.centroid.coords[0]]


def intersection(shape_a, shape_b):
    """Given two shapely shapes, return the shape describing the intersection of them"""
    if shape_a.is_valid and shape_b.is_valid:
        shape_intersection = shape_a.intersection(shape_b)
    else:
        shape_intersection = shape_a.buffer(0).intersection(shape_b.buffer(0))
    return shape_intersection


def union_shape_list(shape_list):
    """Given a list of shapely shapes, return the union of those shapes as one shapely shape"""
    union_shape = shape_list[0]
    for one_shape in shape_list[1:]:
        union_shape = union_shape.union(one_shape)
    return union_shape


def get_overlapping_segment_idxs(geometry, df, spatial_index):
    """Given a shapely shape (geometry) and a geopandas dataframe (df) (and the spatial index of that geopdandas
    dataframe (spatial_index) get the indices corresponding to entries in the geopandas dataframe which over lap the
    shapely shape and return them as a list"""
    possible_matches_index = list(spatial_index.intersection(geometry.bounds))
    possible_matches = df.iloc[possible_matches_index]
    precise_matches = possible_matches[(possible_matches.overlaps(geometry)) | (possible_matches.contains(geometry)) | (
        possible_matches.within(geometry))]
    return list(precise_matches.index.values)


def get_training_polygon_idxs(len_list, training_ratio=0.6):
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    solver.Init(len_list, [len_list], [training_ratio * sum(len_list)])
    solver.Solve()
    return [a for a in range(len(len_list)) if solver.BestSolutionContains(a)]


def oversample(X, y):
    """Function to balance the number of classes via an oversampling of the underrepresented classes. Returns a
    concatenation of the data and the additional samples as well as the corresponding labels"""
    counts = [list(y).count(a) for a in set(y)]
    max_count = max(counts)
    idxs = np.arange(X.shape[0])
    for i in range(len(counts)):
        if counts[i] != max_count:
            idxs = np.hstack((idxs, np.random.choice(np.argwhere(y == i).flatten(), max_count - counts[i])))
    return X[idxs], y[idxs]


def flip_augment(X, y):
    """Function which performs data augmentation via a lateral flip of the image data and returns the concatenation of
    the unflipped and flipped data as well as the labels corresponding to these"""
    return np.concatenate((X, np.flip(X, axis=2))).astype(np.float16), np.hstack((y, y))


def use_pca(X, pca_object=None):
    """Function which will perform dimensionality reduction of input data and return the reduced dimension input as
    well as the PCA object (i.e. to save if the same principal components might need used again)"""
    if pca_object is None:
        pca_object = PCA(n_components=3)
        print('Fitting...')
        pca_object.fit((X.reshape(-1, X.shape[-1]))[::int(X.shape[0] * X.shape[1] * X.shape[2] / 10000)])
        print('Transforming...')
        return pca_object.transform(X.reshape(-1, X.shape[-1])).reshape(-1, config.WINDOW_SIZE, config.WINDOW_SIZE,
                                                                        3), pca_object
    else:
        print('Transforming...')
        return pca_object.transform(X.reshape(-1, X.shape[-1])).reshape(-1, config.WINDOW_SIZE, config.WINDOW_SIZE, 3)


def ocnn_module(fold):
    """
    Train a multi-scale CNN (MS-CNN) model and a single-scale OCNN model to perform LU classification
    (see multi_scale_ocnn.py for model details)

    Using the flag e.g. -md models/multi_scale/48 (or whatever window size is instead of 48), the trained models
    will be saved in the -md directory (.h5 file extensions) as well as the entire LU classification output of each
    model as seg_c_p_c_ocnn_k.npy and seg_c_p_c_ms_k.npy for the OCNN and MS-CNN respectively (for each k).

    For a description of how to recover LU predictions from a seg_c_p_c-like .npy file see README.md
    """

    print("\n\n#######################################")
    print(f"PERFORMING {fold}-FOLD CROSS VALIDATION")
    print("#########################################")

    for k in range(fold):
        
        print("\n#####################")
        print(f"PERFORMING FOLD-{k}")
        print("#####################\n")


        # Update land use annotation manager training indexes
        def update_lu_am_training_idxs(lu_am, k_fold_i):
            # Load file paths of all .shp shapefiles in alphabetical order
            file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
            # Set class training indexes in land use annotation manager as empty dictionary
            lu_am.class_training_idxs = {}
            # Loop through shape files in file paths
            for i, path in enumerate(file_paths):
                # Open shape file
                with fiona.open(path) as shapefile:
                    # Get list containing indexes of features in shapefile if feature properties match fold value i
                    is_training = [idx for idx, feature in enumerate(shapefile) if
                                   int(feature['properties']['k_fold_i']) != k_fold_i]
                # Update training index dictionary to contain new training values
                lu_am.class_training_idxs.update({i: is_training})


        def model_train(path):
            # If benchmark and MS model weights files do not exist for current fold
            if not os.path.isfile(os.path.join(config.MODELS_DIRECTORY,
                                                "ocnn_" + str(k) + ".h5")):
                
                print("############################################")
                print("MODEL WEIGHTS NOT FOUND - CREATING NEW DATA")
                print("############################################")

                # Get training input and output from land use annotation manager 
                # using land cover prob container and rgb container as arguments
                X_train, y_train = lu_am.get_training_ip_op(lc_p_container, rgb_container)
                # Perform data augmentation on data by lateral flip and get concat
                # of unflipped and flipped data and labels
                X_train, y_train = flip_augment(X_train, y_train)

                # Balance number of classes by oversampling underrepresented 
                # classes
                X_train, y_train = oversample(X_train, y_train)

                # Generate random sample of training data
                train_sample = np.random.choice(np.arange(X_train.shape[0]), int(X_train.shape[0] * 1.), replace=False)
                
                # Set training data as training sample
                X_train = X_train[train_sample, ..., :3]
                # Set training labels as size of train_sample
                y_train = y_train[train_sample]

                # Create validation data using land use annotation manager with 
                # land class prob container and rgb container
                X_validate, y_validate = lu_am.get_validation_ip_op(lc_p_container, rgb_container)
                # Shuffle validation data
                X_validate, y_validate = shuffle(X_validate, y_validate)
                print("Xtrain/Xvalidate shapes", X_train.shape, X_validate.shape)

                # Set train and validation labels to categorical via one-hot 
                # encoding
                y_train = to_categorical(y_train)
                y_validate = to_categorical(y_validate)

                # Train benchmark OCNN model
                benchmark_model = train_ocnn_model((X_train, y_train), (X_validate, y_validate),
                                                                path,
                                                                "multiscale_" + str(k))
            else:

                print("############################################")
                print("MODEL WEIGHTS FOUND - LOADING DATA")
                print("############################################")

                # Create benchmark and MS model architecture using number of
                # LU classes
                benchmark_model = BenchmarkModel(config.NUMB_LU_CLASSES)

                # Make predictions using both architectures
                benchmark_model.predict(np.ones((1, config.WINDOW_SIZE, config.WINDOW_SIZE, 3)))

                # Load weights from disk for current fold value of K
                benchmark_model.load_weights(os.path.join(config.MODELS_DIRECTORY,
                                                      "ocnn_" + str(k) + ".h5"))

            print("\n################################")
            print(f"RUNNING PREDICTIONS USING MS-CNN")
            print("################################\n")

            # If prediction file for MS-CNN fold K does not exist
            if not os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'seg_c_p_c_ocnn_' + str(k) + '.npy')):
                print("Performing k: %s ocnn predictions..." % str(k))
                # Update segments for OCNN architecture
                segment_container.prob_container.update_segments(benchmark_model, lc_p_container, segment_container,
                                                                rgb_container)
                np.save(os.path.join(config.MODELS_DIRECTORY, 'seg_c_p_c_ocnn_' + str(k) + '.npy'),
                        segment_container.prob_container.probabilities)


        # Run update land use manager function with annotation manager 
        # and fold value K as arguments
        update_lu_am_training_idxs(lu_am, k)

        model_train(config.MODELS_DIRECTORY)

def multiscale_module(fold, runtype=None):
    """
    Train a multi-scale CNN (MS-CNN) model and a single-scale OCNN model to perform LU classification
    (see multi_scale_ocnn.py for model details)

    Using the flag e.g. -md models/multi_scale/48 (or whatever window size is instead of 48), the trained models
    will be saved in the -md directory (.h5 file extensions) as well as the entire LU classification output of each
    model as seg_c_p_c_ocnn_k.npy and seg_c_p_c_ms_k.npy for the OCNN and MS-CNN respectively (for each k).

    For a description of how to recover LU predictions from a seg_c_p_c-like .npy file see README.md
    """

    print("\n\n#######################################")
    print(f"PERFORMING {fold}-FOLD CROSS VALIDATION")
    print("#########################################")

    for k in range(fold):
        
        print("\n#####################")
        print(f"PERFORMING FOLD-{k}")
        print("#####################\n")


        # Update land use annotation manager training indexes
        def update_lu_am_training_idxs(lu_am, k_fold_i):
            # Load file paths of all .shp shapefiles in alphabetical order
            file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
            # Set class training indexes in land use annotation manager as empty dictionary
            lu_am.class_training_idxs = {}
            # Loop through shape files in file paths
            for i, path in enumerate(file_paths):
                # Open shape file
                with fiona.open(path) as shapefile:
                    # Get list containing indexes of features in shapefile if feature properties match fold value i
                    is_training = [idx for idx, feature in enumerate(shapefile) if
                                   int(feature['properties']['k_fold_i']) != k_fold_i]
                # Update training index dictionary to contain new training values
                lu_am.class_training_idxs.update({i: is_training})


        def model_train(path):
            # If benchmark and MS model weights files do not exist for current fold
            if not (os.path.isfile(os.path.join(path,
                                                "multiscale_" + str(k) + ".h5"))):
                
                print("############################################")
                print("MODEL WEIGHTS NOT FOUND - CREATING NEW DATA")
                print("############################################")

                # Get training input and output from land use annotation manager 
                # using land cover prob container and rgb container as arguments
                X_train, y_train = lu_am.get_training_ip_op(lc_p_container, rgb_container)
                # Perform data augmentation on data by lateral flip and get concat
                # of unflipped and flipped data and labels
                X_train, y_train = flip_augment(X_train, y_train)

                # Balance number of classes by oversampling underrepresented 
                # classes
                X_train, y_train = oversample(X_train, y_train)

                # Generate random sample of training data
                train_sample = np.random.choice(np.arange(X_train.shape[0]), int(X_train.shape[0] * 1.), replace=False)
                
                # Set training data as training sample
                X_train = X_train[train_sample, ..., :3]
                # Set training labels as size of train_sample
                y_train = y_train[train_sample]

                # Create validation data using land use annotation manager with 
                # land class prob container and rgb container
                X_validate, y_validate = lu_am.get_validation_ip_op(lc_p_container, rgb_container)
                # Shuffle validation data
                X_validate, y_validate = shuffle(X_validate, y_validate)
                print("Xtrain/Xvalidate shapes", X_train.shape, X_validate.shape)

                # Set train and validation labels to categorical via one-hot 
                # encoding
                y_train = to_categorical(y_train)
                y_validate = to_categorical(y_validate)

                # Train benchmark OCNN and MS-CNN models
                #benchmark_model, 
                ms_model = train_multiscale_model((X_train, y_train), (X_validate, y_validate),
                                                                path,
                                                                "multiscale_" + str(k))
            else:

                print("############################################")
                print("MODEL WEIGHTS FOUND - LOADING DATA")
                print("############################################")

                # Create benchmark and MS model architecture using number of
                # LU classes
                ms_model = MultiScaleModel(config.NUMB_LU_CLASSES)

                # Make predictions using both architectures
                ms_model.predict(np.ones((1, config.WINDOW_SIZE, config.WINDOW_SIZE, 3)))

                # Load weights from disk for current fold value of K
                ms_model.load_weights(os.path.join(path,
                                                "multiscale_" + str(k) + ".h5"))

            print("\n################################")
            print(f"RUNNING PREDICTIONS USING MS-CNN")
            print("################################\n")

            # If prediction file for MS-CNN fold K does not exist
            if not os.path.isfile(os.path.join(path, 'seg_c_p_c_ms_' + str(k) + '.npy')):
                print("Performing k: %s ms-cnn predictions..." % str(k))
                # Update segments for MS architecture
                segment_container.prob_container.update_segments(ms_model, lc_p_container, segment_container,
                                                                rgb_container)
                np.save(os.path.join(path, 'seg_c_p_c_ms_' + str(k) + '.npy'),
                        segment_container.prob_container.probabilities)


        # Run update land use manager function with annotation manager 
        # and fold value K as arguments
        update_lu_am_training_idxs(lu_am, k)

        model_train(config.MODELS_DIRECTORY)


def adaptive_multiscale_module(fold, runtype=None, adaptivefold=None):
    """
    Train a multi-scale CNN (MS-CNN) model to perform LU classification
    (see multi_scale_ocnn.py for model details)

    Using the flag e.g. -md models/multi_scale/48 (or whatever window size is instead of 48), the trained models
    will be saved in the -md directory (.h5 file extensions) as well as the entire LU classification output of each
    model as seg_c_p_c_ocnn_k.npy and seg_c_p_c_ms_k.npy for the OCNN and MS-CNN respectively (for each k).

    For a description of how to recover LU predictions from a seg_c_p_c-like .npy file see README.md
    """

    print("\n\n#######################################")
    print(f"PERFORMING {fold}-FOLD CROSS VALIDATION")
    print("#########################################")

    for k in range(fold):
        
        print("\n#####################")
        print(f"PERFORMING FOLD-{k}")
        print("#####################\n")


        # Update land use annotation manager training indexes
        def update_lu_am_training_idxs(lu_am, k_fold_i):
            # Load file paths of all .shp shapefiles in alphabetical order
            file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
            # Set class training indexes in land use annotation manager as empty dictionary
            lu_am.class_training_idxs = {}
            # Loop through shape files in file paths
            for i, path in enumerate(file_paths):
                # Open shape file
                with fiona.open(path) as shapefile:
                    # Get list containing indexes of features in shapefile if feature properties match fold value i
                    is_training = [idx for idx, feature in enumerate(shapefile) if
                                   int(feature['properties']['k_fold_i']) != k_fold_i]
                # Update training index dictionary to contain new training values
                lu_am.class_training_idxs.update({i: is_training})


        def model_train(path):
            # If benchmark and MS model weights files do not exist for current fold
            if not (os.path.isfile(os.path.join(path,
                                                "multiscale_" + str(k) + ".h5"))):
                
                print("############################################")
                print("MODEL WEIGHTS NOT FOUND - CREATING NEW DATA")
                print("############################################")

                # Get training input and output from land use annotation manager 
                # using land cover prob container and rgb container as arguments
                X_train, y_train = lu_am.get_training_ip_op(lc_p_container, rgb_container)
                # Perform data augmentation on data by lateral flip and get concat
                # of unflipped and flipped data and labels
                X_train, y_train = flip_augment(X_train, y_train)

                # Balance number of classes by oversampling underrepresented 
                # classes
                X_train, y_train = oversample(X_train, y_train)

                # Generate random sample of training data
                train_sample = np.random.choice(np.arange(X_train.shape[0]), int(X_train.shape[0] * 1.), replace=False)
                
                # Set training data as training sample
                X_train = X_train[train_sample, ..., :3]
                # Set training labels as size of train_sample
                y_train = y_train[train_sample]

                # Create validation data using land use annotation manager with 
                # land class prob container and rgb container
                X_validate, y_validate = lu_am.get_validation_ip_op(lc_p_container, rgb_container)
                # Shuffle validation data
                X_validate, y_validate = shuffle(X_validate, y_validate)
                print("Xtrain/Xvalidate shapes", X_train.shape, X_validate.shape)

                # Set train and validation labels to categorical via one-hot 
                # encoding
                y_train = to_categorical(y_train)
                y_validate = to_categorical(y_validate)

                # Train benchmark OCNN and MS-CNN models
                #benchmark_model, 
                ms_model = train_adaptive_multiscale_model((X_train, y_train), (X_validate, y_validate),
                                                                path,
                                                                "multiscale_" + str(k))
            else:

                print("############################################")
                print("MODEL WEIGHTS FOUND - LOADING DATA")
                print("############################################")

                # Create MS model architecture using number of
                # LU classes
                ms_model = MultiScaleModel(config.NUMB_LU_CLASSES)

                # Make predictions using both architectures
                ms_model.predict(np.ones((1, config.WINDOW_SIZE, config.WINDOW_SIZE, 3)))

                # Load weights from disk for current fold value of K
                ms_model.load_weights(os.path.join(path,
                                                "multiscale_" + str(k) + ".h5"))

            print("\n################################")
            print(f"RUNNING PREDICTIONS USING MS-CNN")
            print("################################\n")

            # If prediction file for MS-CNN fold K does not exist
            if not os.path.isfile(os.path.join(path, 'seg_c_p_c_ms_' + str(k) + '.npy')):
                print("Performing k: %s ms-cnn predictions..." % str(k))
                # Update segments for MS architecture
                segment_container.prob_container.update_segments(ms_model, lc_p_container, segment_container,
                                                                rgb_container)
                np.save(os.path.join(path, 'seg_c_p_c_ms_' + str(k) + '.npy'),
                        segment_container.prob_container.probabilities)


        # Run update land use manager function with annotation manager 
        # and fold value K as arguments
        update_lu_am_training_idxs(lu_am, k)

        if runtype == None: path = config.MODELS_DIRECTORY
        elif "adaptive" in runtype:
            type = runtype.split("-")[-1]
            if type == "full": 
                path = os.path.join(config.MODELS_DIRECTORY, 'new_data',  str(config.WINDOW_SIZE))
            elif type in ("0", "1", "2", "3"): 
                path = os.path.join(config.MODELS_DIRECTORY, 'temp_data',  str(config.WINDOW_SIZE), str(int(type)))

        model_train(path)

def pixelwise_CNN(numb_folds=5):
    """
    Train a pixelwise CNN model to perform LU classification
    (see BenchmarkModel in multi_scale_ocnn.py for model details)

    Using the flag e.g. -md models/pixelwise_CNN/48 (or whatever window size is instead of 48), the trained models
    will be saved in the -md directory (.h5 file extensions) as well as some LU classification outputs corresponding
    to validation annotations for each k (the reason why a complete LU output is not generated as is done for
    multiscale() is that the pixelwise CNN inference time is MUCH slower so only some pixels are classified).
    """

    print("###################")
    print(f"PERFORMING {numb_folds}-FOLD CROSS VALIDATION")
    print("###################")

    def get_one_tile_pixel_idxs_from_pixel_idx(pixel_idx):
        i, j = divmod(pixel_idx, config.TIF.shape[1])
        i, j = int(i), int(j)
        i_list = [i_shift + i for i_shift in range(-int(config.WINDOW_SIZE / 2), int(config.WINDOW_SIZE / 2))]
        i_list = [a if 0 <= a < config.TIF.shape[0] else -1 for a in i_list]
        j_list = [j_shift + j for j_shift in range(-int(config.WINDOW_SIZE / 2), int(config.WINDOW_SIZE / 2))]
        j_list = [a if 0 <= a < config.TIF.shape[1] else -1 for a in j_list]
        return np.array(
            [config.TIF.shape[1] * each_i + each_j if each_i != -1 and each_j != -1 else -1 for each_i in i_list for
             each_j
             in j_list], dtype=np.int32)

    def get_training_ip_op(rgb_c_object, k_fold_i, samples=3000):
        file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
        model_input, model_output = [], []
        for i, path in tqdm(enumerate(file_paths)):
            class_pixel_idxs = []
            with fiona.open(path) as shapefile:
                is_training = [idx for idx, feature in enumerate(shapefile) if
                               int(feature['properties']['k_fold_i']) != k_fold_i]
                shapes = [shape(shapefile[idx]['geometry']) for idx in is_training]
                for poly in shapes:
                    class_pixel_idxs.extend(get_shape_pixel_idxs(poly))
            class_pixel_idxs = list(set(class_pixel_idxs))
            training_samples = np.random.choice(class_pixel_idxs, min(len(class_pixel_idxs), samples),
                                                replace=False)
            for pixel_idx in training_samples:
                model_input.append(
                    rgb_c_object.get_many_rgb(get_one_tile_pixel_idxs_from_pixel_idx(pixel_idx)).reshape(
                        config.WINDOW_SIZE, config.WINDOW_SIZE, 3))
            model_output.extend([i] * (len(model_input) - len(model_output)))
        return np.array(model_input, dtype=np.float32), np.array(model_output, np.uint8)

    def get_validation_ip_op(rgb_c_object, k_fold_i, samples=3000):
        file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
        model_input, model_output = [], []
        for i, path in tqdm(enumerate(file_paths)):
            class_pixel_idxs = []
            with fiona.open(path) as shapefile:
                is_validation = [idx for idx, feature in enumerate(shapefile) if
                                 int(feature['properties']['k_fold_i']) == k_fold_i]
                shapes = [shape(shapefile[idx]['geometry']) for idx in is_validation]
                for poly in shapes:
                    class_pixel_idxs.extend(get_shape_pixel_idxs(poly))
            class_pixel_idxs = list(set(class_pixel_idxs))
            training_samples = np.random.choice(class_pixel_idxs, min(len(class_pixel_idxs), samples),
                                                replace=False)
            for pixel_idx in training_samples:
                model_input.append(
                    rgb_c_object.get_many_rgb(get_one_tile_pixel_idxs_from_pixel_idx(pixel_idx)).reshape(
                        config.WINDOW_SIZE, config.WINDOW_SIZE, 3))
            model_output.extend([i] * (len(model_input) - len(model_output)))
        return np.array(model_input, dtype=np.float32), np.array(model_output, np.uint8)

    def evaluate(model, rgb_c_object, k_fold_i, chunk_size=1024 * 10):
        print("Evaluating pixelwise CNN model...")
        file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
        pixel_idx_map = []
        lu_probabilities = []
        for i, path in tqdm(enumerate(file_paths)):
            class_pixel_idxs = []
            with fiona.open(path) as shapefile:
                is_validation = [idx for idx, feature in enumerate(shapefile) if
                                 int(feature['properties']['k_fold_i']) == k_fold_i]
                shapes = [shape(shapefile[idx]['geometry']) for idx in is_validation]
                for poly in shapes:
                    class_pixel_idxs.extend(get_shape_pixel_idxs(poly))
            class_pixel_idxs = list(set(class_pixel_idxs))
            pixel_idx_map.extend(class_pixel_idxs)
            for chunk_numb in tqdm(range(len(class_pixel_idxs) // chunk_size + 1)):
                numb_tiles = chunk_size if chunk_numb != len(class_pixel_idxs) // chunk_size else len(
                    class_pixel_idxs) % chunk_size
                chunk_pixels = [idx for z, idx in enumerate(class_pixel_idxs) if
                                chunk_numb * chunk_size <= z < chunk_numb * chunk_size + numb_tiles]
                model_input = []
                for idx in chunk_pixels:
                    model_input.append(rgb_c_object.get_many_rgb(get_one_tile_pixel_idxs_from_pixel_idx(idx)).reshape(
                        config.WINDOW_SIZE, config.WINDOW_SIZE, 3))
                model_input = np.stack(model_input)
                model_output = model.predict(model_input)
                lu_probabilities.append(model_output)
        print("Saving...")
        lu_probabilities = np.concatenate(lu_probabilities)
        np.save(
            os.path.join(config.MODELS_DIRECTORY, 'pixel_probabilities_pixelwise_' + str(k) + '.npy'),
            lu_probabilities)
        np.save(
            os.path.join(config.MODELS_DIRECTORY, "pixel_idx_map_" + str(k_fold_i) + ".npy"),
            np.array(pixel_idx_map))

    for k in range(numb_folds):  # for each k-fold cross val (if just one result needed you can change to range(1))
        
        print("#####################")
        print(f"PERFORMING FOLD-{k}")
        print("#####################\n")
        
        print("\n#######################")
        print("GENERATING TRAINING DATA")
        print("#########################")
        X_train, y_train = get_training_ip_op(rgb_container, 0, 3000)  # 3000 is numb training samples chosen
        X_validate, y_validate = get_validation_ip_op(rgb_container, 0, 750)  # 750 is numb validation samples chosen
        y_train = to_categorical(y_train)
        y_validate = to_categorical(y_validate)
        print("X_train/X_validate shapes", X_train.shape, X_validate.shape)


        benchmark_model = BenchmarkModel(y_train.shape[-1])
        benchmark_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                                loss=losses.categorical_crossentropy,
                                metrics=['accuracy'],
                                )
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
        mc = ModelCheckpoint(
            os.path.join(config.MODELS_DIRECTORY, "pixelwise_" + str(k) + ".h5"),
            monitor='val_loss', mode='min', verbose=0, save_best_only=True,
            save_weights_only=False)
        benchmark_model.fit(X_train, y_train, epochs=100,
                            validation_data=(X_validate, y_validate),
                            callbacks=[es, mc], batch_size=64)

        evaluate(benchmark_model, rgb_container, k)

def dual_mscnn_ocnn(fold, runtype=None):
    """
    Train a multi-scale CNN (MS-CNN) model and a single-scale OCNN model to perform LU classification
    (see multi_scale_ocnn.py for model details)

    Using the flag e.g. -md models/multi_scale/48 (or whatever window size is instead of 48), the trained models
    will be saved in the -md directory (.h5 file extensions) as well as the entire LU classification output of each
    model as seg_c_p_c_ocnn_k.npy and seg_c_p_c_ms_k.npy for the OCNN and MS-CNN respectively (for each k).

    For a description of how to recover LU predictions from a seg_c_p_c-like .npy file see README.md
    """

    print("\n\n#######################################")
    print(f"PERFORMING {fold}-FOLD CROSS VALIDATION")
    print("#########################################")

    for k in range(fold):
        
        print("\n#####################")
        print(f"PERFORMING FOLD-{k}")
        print("#####################\n")


        # Update land use annotation manager training indexes
        def update_lu_am_training_idxs(lu_am, k_fold_i):
            # Load file paths of all .shp shapefiles in alphabetical order
            file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
            # Set class training indexes in land use annotation manager as empty dictionary
            lu_am.class_training_idxs = {}
            # Loop through shape files in file paths
            for i, path in enumerate(file_paths):
                # Open shape file
                with fiona.open(path) as shapefile:
                    # Get list containing indexes of features in shapefile if feature properties match fold value i
                    is_training = [idx for idx, feature in enumerate(shapefile) if
                                   int(feature['properties']['k_fold_i']) != k_fold_i]
                # Update training index dictionary to contain new training values
                lu_am.class_training_idxs.update({i: is_training})


        def model_train(path):
            # If benchmark and MS model weights files do not exist for current fold
            if not (os.path.isfile(os.path.join(path,
                                                "multiscale_" + str(k) + ".h5"))):
                
                print("############################################")
                print("MODEL WEIGHTS NOT FOUND - CREATING NEW DATA")
                print("############################################")

                # Get training input and output from land use annotation manager 
                # using land cover prob container and rgb container as arguments
                X_train, y_train = lu_am.get_training_ip_op(lc_p_container, rgb_container)
                # Perform data augmentation on data by lateral flip and get concat
                # of unflipped and flipped data and labels
                X_train, y_train = flip_augment(X_train, y_train)

                # Balance number of classes by oversampling underrepresented 
                # classes
                X_train, y_train = oversample(X_train, y_train)

                # Generate random sample of training data
                train_sample = np.random.choice(np.arange(X_train.shape[0]), int(X_train.shape[0] * 1.), replace=False)
                
                # Set training data as training sample
                X_train = X_train[train_sample, ..., :3]
                # Set training labels as size of train_sample
                y_train = y_train[train_sample]

                # Create validation data using land use annotation manager with 
                # land class prob container and rgb container
                X_validate, y_validate = lu_am.get_validation_ip_op(lc_p_container, rgb_container)
                # Shuffle validation data
                X_validate, y_validate = shuffle(X_validate, y_validate)
                print("Xtrain/Xvalidate shapes", X_train.shape, X_validate.shape)

                # Set train and validation labels to categorical via one-hot 
                # encoding
                y_train = to_categorical(y_train)
                y_validate = to_categorical(y_validate)

                # Train benchmark OCNN and MS-CNN models
                #benchmark_model, 
                benchmark_model, ms_model = train_dual_multiscale_model((X_train, y_train), (X_validate, y_validate),
                                                                path,
                                                                "multiscale_" + str(k))
            else:

                print("############################################")
                print("MODEL WEIGHTS FOUND - LOADING DATA")
                print("############################################")

                # Create benchmark and MS model architecture using number of
                # LU classes
                ms_model = MultiScaleModel(config.NUMB_LU_CLASSES)
                benchmark_model = BenchmarkModel(config.NUMB_LU_CLASSES)
                
                # Make predictions using both architectures
                ms_model.predict(np.ones((1, config.WINDOW_SIZE, config.WINDOW_SIZE, 3)))
                benchmark_model.predict(np.ones((1, config.WINDOW_SIZE, config.WINDOW_SIZE, 3)))
                # Load weights from disk for current fold value of K
                benchmark_model.load_weights(os.path.join(config.MODELS_DIRECTORY,
                                                      "multiscale_" + str(k) + "_benchmark.h5"))
                ms_model.load_weights(os.path.join(path,
                                                "multiscale_" + str(k) + ".h5"))

            print("\n################################")
            print(f"RUNNING PREDICTIONS USING MS-CNN")
            print("################################\n")

            # If prediction file for MS-CNN fold K does not exist
            if not os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'seg_c_p_c_ocnn_' + str(k) + '.npy')):
                print("Performing k: %s ocnn predictions..." % str(k))
                # Update segments for OCNN architecture
                segment_container.prob_container.update_segments(benchmark_model, lc_p_container, segment_container,
                                                                rgb_container)
                np.save(os.path.join(config.MODELS_DIRECTORY, 'seg_c_p_c_ocnn_' + str(k) + '.npy'),
                        segment_container.prob_container.probabilities)

            print("\n################################")
            print(f"RUNNING PREDICTIONS USING MS-CNN")
            print("################################\n")

            # If prediction file for MS-CNN fold K does not exist
            if not os.path.isfile(os.path.join(path, 'seg_c_p_c_ms_' + str(k) + '.npy')):
                print("Performing k: %s ms-cnn predictions..." % str(k))
                # Update segments for MS architecture
                segment_container.prob_container.update_segments(ms_model, lc_p_container, segment_container,
                                                                rgb_container)
                np.save(os.path.join(path, 'seg_c_p_c_ms_' + str(k) + '.npy'),
                        segment_container.prob_container.probabilities)


        # Run update land use manager function with annotation manager 
        # and fold value K as arguments
        update_lu_am_training_idxs(lu_am, k)

        model_train(config.MODELS_DIRECTORY)

def obia_svm(numb_folds=5):
    """
    Train an OBIA-SVM to perform LU classification

    Using the flag e.g. -md models/obia_svm, the trained models will be saved in the -md directory (.pkl file
    extension) as well as the entire LU classification output of each model as seg_c_p_c_obia_svm_k.npy (for each k).

    For a description of how to recover LU predictions from a seg_c_p_c-like .npy file see README.md
    """

    print("###############################################")
    print(f"PERFORMING {numb_folds}-FOLD CROSS VALIDATION")
    print("###############################################")

    def get_training_ip_op(rgb_c_object, k_fold_i):
        file_paths = sorted([path for path in glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp')])  # alphabetical
        model_input, model_output = [], []
        for i, path in tqdm(enumerate(file_paths)):
            with fiona.open(path) as shapefile:
                is_training = [idx for idx, feature in enumerate(shapefile) if
                               int(feature['properties']['k_fold_i']) != k_fold_i]
                shapes = [shape(shapefile[idx]['geometry']) for idx in is_training if
                          shape(shapefile[idx]['geometry']).is_valid]
            annotation_shapes = union_shape_list(shapes)
            object_gdf = gpd.read_file(config.SEGMENTATION_SHAPEFILE)
            training_objects = object_gdf.loc[object_gdf.overlaps(annotation_shapes)]
            for one_shape in training_objects.geometry.values:
                one_shape_data = rgb_c_object.get_many_rgb(get_shape_pixel_idxs(one_shape))
                if one_shape_data.shape[0] > 1:
                    features = []
                    for channel in range(one_shape_data.shape[1]):
                        description = describe(one_shape_data[:, channel])
                        description = list(description.minmax) + list(description)[2:]
                        features += description
                    model_input.append(features)
                    model_output.append(i)
        return np.array(model_input, dtype=np.float32), np.array(model_output, np.uint8)

    ## SVM ERROR CURRENTLY LOCATED IN THIS FUNCTION - STILL UNSURE OF ORIGIN
    # ANYTHING COMMENTED OUT IN CODE IS TO AID IN DEBUGGING AND FINDING ERROR
    def update_segments(model, seg_c_object, rgb_c_object):
        with np.errstate(divide='ignore', invalid='ignore'):
            chunk_size = 1024
            segments_gdf = gpd.read_file(config.SEGMENTATION_SHAPEFILE)
            seg_shape_list = list(segments_gdf.geometry.values)
            for i in tqdm(range(config.NUMB_SEGMENTS // chunk_size + 1)):
                
                model_input = []
                numb_tiles = chunk_size if i != config.NUMB_SEGMENTS // chunk_size else config.NUMB_SEGMENTS % chunk_size
                chunk_segment_indexes = [a for a in range(i * chunk_size, i * chunk_size + numb_tiles)]
                
                for segment_index in chunk_segment_indexes:
                    segment_pixel_idxs = get_shape_pixel_idxs(seg_shape_list[segment_index])
                    one_shape_data = rgb_c_object.get_many_rgb(segment_pixel_idxs)
                    
                    if one_shape_data.shape[0] == 1:
                        np.vstack((one_shape_data, one_shape_data))

                    features = []

                    for channel in range(one_shape_data.shape[1]):
                        
                        # Potentially uneeded, maybe just supress warnings
                        # if len(one_shape_data[:, channel]) == 1:
                        #     n = random.randint(0, 5)
                        #     item = one_shape_data[:, channel][0]
                        #     for i in range(n):
                        #         np.append(one_shape_data[:, channel], item)
                        # if len(one_shape_data[:, channel])==1:
                        #     one_shape_data[:, channel] = [one_shape_data[:, channel], one_shape_data[:, channel]]

                        description = describe(one_shape_data[:, channel]) # Pretty sure it's this line
                        description = list(description.minmax) + list(description)[2:]
                        features += description

                    model_input.append(np.nan_to_num(features, nan=0, posinf=33333333, neginf=33333333)) # Not this line
                    
                model_input = np.array(model_input, dtype=np.float32) # Not this line
                model_output = model.predict_proba(model_input) * np.iinfo(np.uint16).max # Not this line
                seg_c_object.prob_container.probabilities[chunk_segment_indexes] = model_output # Not this line
            

    for k in range(numb_folds):  # for each k-fold cross val (if just one result needed you can change to range(1))
        print("\n#####################")
        print(f"PERFORMING FOLD-{k}")
        print("#####################\n")
        
        print("\n#######################")
        print("GENERATING TRAINING DATA")
        print("#########################")
        X_train, y_train = get_training_ip_op(rgb_container, k)


        print("\n#######################")
        print("CREATING AND FITTING SVM")
        print("#########################")
        obia_svm_model = SVC(kernel='rbf', probability=True)  # radial basis function (rbf) kernel
        obia_svm_model.fit(X_train, y_train)


        print("\n#####################")
        print("UPDATING SEGMENTS DATA")
        print("#######################")
        update_segments(obia_svm_model, segment_container, rgb_container)
        
        print("\n#######################")
        print("SAVING GENERATED DATA TO DISK")
        print("#########################")
        np.save(os.path.join(config.MODELS_DIRECTORY, "seg_c_p_c_obia_svm_" + str(k) + ".npy"),
                segment_container.prob_container.probabilities)
        with open(os.path.join(config.MODELS_DIRECTORY, "obia_svm_model_" + str(k) + ".pkl"), "wb") as f:
            pickle.dump(obia_svm_model, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--models_directory", help="path to models directory e.g. models/JDLs/31032019_0")
    parser.add_argument("-tp", "--tif_path", help="path to data .tif",
                        default="geo_data/K_R3C4/NI_Kano_20Q1_V0_R3C4.tif")
    parser.add_argument("-lc", "--lc_annotations",
                        help="path to directory containing lc annotations e.g. geo_data/K_R3C4/lc_annotations",
                        default='geo_data/K_R3C4/lc_annotations')
    parser.add_argument("-lu", "--lu_annotations",
                        help="path to directory containing lu annotations e.g. geo_data/K_R3C4/lu_annotations",
                        default='geo_data/K_R3C4/lu_annotations')
    parser.add_argument("-s", "--segmentation_file",
                        help="path to segmentation shapefile e.g. geo_data/K_R3C4/R3C430_8_196_full_tif.shp",
                        default='geo_data/K_R3C4/R3C430_8_196_full_tif.shp')
    parser.add_argument("-ws", "--window_size", help="number of pixels used as window height/width", default=96)
    parser.add_argument("-au", "--auto", help="Model number used for automatic code running", default=0)
    args = parser.parse_args()

    """ PATH SPECIFICATION VALIDATION """
    if not os.path.isdir(args.models_directory):
        print("Specified model directory does not exist, creating")
        os.mkdir(args.models_directory)
    if not os.path.isfile(args.tif_path):
        print("Cannot find specified .tif file, please check and restart")
        quit()
    if not os.path.isdir(args.lc_annotations):
        print("Cannot find specified LC annotation directory, please check and restart")
        quit()
    if not os.path.isdir(args.lu_annotations):
        print("Cannot find specified LU annotation directory, please check and restart")
        quit()
    if not os.path.isfile(args.segmentation_file):
        print("Cannot find specified segmentation shapefile, please check and restart")
        quit()

    config.TIF_PATH = args.tif_path
    config.TIF = rasterio.open(config.TIF_PATH)
    config.IDXS_TIF = make_idxs_tif()
    config.MODELS_DIRECTORY = args.models_directory
    config.LC_ANNOTATION_DIRECTORY = args.lc_annotations
    config.LU_ANNOTATION_DIRECTORY = args.lu_annotations
    config.SEGMENTATION_SHAPEFILE = args.segmentation_file
    config.NUMB_LC_CLASSES = len(glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp'))
    config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))
    config.WINDOW_SIZE = int(args.window_size)

    print("##########################")
    print("CONFIRMING GPUS AVAILABLE")
    print("##########################")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    print("\n###########################################")
    print("LOADING ANNOTATION MANAGERS AND CONTAINERS")
    print("###########################################\n")
    start = time()

    if os.path.isfile(config.MODELS_DIRECTORY + '/lc_am.pkl'):
        with open(config.MODELS_DIRECTORY + '/lc_am.pkl', 'rb') as f:
            lc_am = pickle.load(f)
    else:
        print("\n#################################################")
        print("LAND COVER ANNOTATION MANAGER NOT FOUND - CREATING")
        print("###################################################\n")
        lc_am = LandCoverAnnotationManager()

    if os.path.isfile(config.MODELS_DIRECTORY + '/lu_am.pkl'):
        with open(config.MODELS_DIRECTORY + '/lu_am.pkl', 'rb') as f:
            lu_am = pickle.load(f)
    else:
        print("\n###############################################")
        print("LAND USE ANNOTATION MANAGER NOT FOUND - CREATING")
        print("#################################################\n")
        lu_am = LandUseAnnotationManager()

    if os.path.isfile(config.MODELS_DIRECTORY + '/rgb_c.pkl'):
        with open(config.MODELS_DIRECTORY + '/rgb_c.pkl', 'rb') as f:
            rgb_container = pickle.load(f)
    else:
        print("\n#################################")
        print("RGB CONTAINER NOT FOUND - CREATING")
        print("###################################\n")
        rgb_container = RGBContainer()

    if os.path.isfile(config.MODELS_DIRECTORY + '/seg_c.pkl'):
        with open(config.MODELS_DIRECTORY + '/seg_c.pkl', 'rb') as f:
            segment_container = pickle.load(f)
    else:
        print("\n#####################################")
        print("SEGMENT CONTAINER NOT FOUND - CREATING")
        print("#######################################\n")
        segment_container = SegmentContainer()

    config.NUMB_SEGMENTS = segment_container.prob_container.probabilities.shape[0]
    # Create LC Prob container using segment container, land class annotation 
    # manager and land use annotation manager
    lc_p_container = LCProbabilityContainer(segment_container, lc_am, lu_am)
    
    end = time()

    print('Time taken (initiation): %s' % (end - start))
    print("Window size: %s" % str(config.WINDOW_SIZE))


    def choice_input():
        print("\n###############################")
        print("PLEASE CHOSE A MODEL TYPE TO RUN")
        print("#################################\n")
        print("1. Multi-Scale CNN")
        print("2. OCNN Benchmark")
        print("3. Pixelwise CNN")
        print("4. OBIA SVM\n")
        print("5. MS-CNN and OCNN Benchmark Dual Mode")
        choice = input("Please enter one of the numbers above as an option (enter q to quit): ")
        print("")

        if choice in ("1", "2", "3", "4", "5"):
            choice = int(choice)
            if choice == 1:
                fold = int(input("How many folds should be performed in Multi-Scale K-Fold Cross Validation (Integers > 0 Only)? "))
                fold = fold if fold > 0 else 1
                multiscale_module(fold)
            elif choice == 2:
                fold = int(input("How many folds should be performed in OCNN K-Fold Cross Validation (Integers > 0 Only)? "))
                fold = fold if fold > 0 else 1
                ocnn_module(fold)
            elif choice == 3:
                numb_folds = int(input("How many folds should be performed in PixelWise K-Fold Cross Validation (Integers > 0 Only)? "))
                numb_folds = numb_folds if numb_folds > 0 else 1
                pixelwise_CNN(numb_folds)
            elif choice == 4:
                numb_folds = int(input("How many folds should be performed in OBIA-SVM K-Fold Cross Validation (Integers > 0 Only)? "))
                numb_folds = numb_folds if numb_folds > 0 else 1
                obia_svm(numb_folds)
            elif choice == 5:
                numb_folds = int(input("How many folds should be performed in dual MSCNN OCNN K-Fold Cross Validation (Integers > 0 Only)? "))
                numb_folds = numb_folds if numb_folds > 0 else 1
                dual_mscnn_ocnn(numb_folds)
        elif choice == "q":
            quit()
        else:
            print("That is not a valid option, please enter a valid option")
            choice_input()

    auto_val = int(args.auto)

    if(auto_val != 0 and isinstance(auto_val, int) and auto_val in (1, 2, 3, 4, 5)):
        choice = auto_val
        if choice == 1:
            print("\n##############################")
            print("RUNNING AUTOMATIC 5-FOLD MS-CNN")
            print("################################\n")
            multiscale_module(5)
        elif choice == 2:
            print("\n#####################################")
            print("RUNNING AUTOMATIC 5-FOLD OCNN")
            print("#######################################\n")
            ocnn_module(5)
        elif choice == 3:
            print("\n#####################################")
            print("RUNNING AUTOMATIC 5-FOLD Pixelwise-CNN")
            print("#######################################\n")
            pixelwise_CNN(5)
        elif choice == 4:
            print("\n################################")
            print("RUNNING AUTOMATIC 5-FOLD OBIA-SVM")
            print("##################################\n")
            obia_svm(5)
        elif choice == 4:
            print("\n################################")
            print("RUNNING AUTOMATIC 5-FOLD DUAL MSCNN AND OCNN")
            print("##################################\n")
            dual_mscnn_ocnn(5)
    elif auto_val == 0:
        choice_input()
    else:
        print("Error with automatic code parameter, please check")
        quit()
