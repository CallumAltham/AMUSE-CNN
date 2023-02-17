# Check all module imports for those that are used

from typing import ClassVar
import housekeeping
import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rasterio.plot import show
from shapely.geometry import box, shape
import numpy as np
import fiona
import rasterio
import tensorflow as tf
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.lines as mlines
import geopandas as gpd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
import os
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from shapely.geometry import MultiPolygon
import json
import pickle
from itertools import cycle
from results import close_fig, oa_from_confusion_matrix, kappa_from_confusion_matrix
from land_classes import LandCoverClasses, LandUseClasses

from run import LCProbabilityContainer, LUProbabilityContainer, LandUseAnnotationManager, \
    LandCoverAnnotationManager, LandCoverPolygon, LandUsePolygon, SegmentContainer, get_shape_pixel_idxs, \
    RGBContainer
import run
import ocnns, mlps

import config

lines = ["-", "--", "-.", ":"]
line_cycle = cycle(lines)


def load_files(runtype=None):
    if runtype == None:
        with open(config.MODELS_DIRECTORY + '/rgb_c.pkl', 'rb') as f:
            rgb_container = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/lc_am.pkl', 'rb') as f:
            lc_am_object = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/lu_am.pkl', 'rb') as f:
            lu_am_object = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/seg_c.pkl', 'rb') as f:
            segment_container = pickle.load(f)
            lc_p_c = LCProbabilityContainer(segment_container, lc_am_object, lu_am_object)
    elif runtype == "adaptive":
        with open(config.MODELS_DIRECTORY + '/temp_data/' + str(config.WINDOW_SIZE) + f'/{str(config.ADAPTIVE_RUN)}/rgb_c_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'rb') as f:
            rgb_container = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/temp_data/' + str(config.WINDOW_SIZE) + f'/{str(config.ADAPTIVE_RUN)}/lc_am_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'rb') as f:
            lc_am_object = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/temp_data/' + str(config.WINDOW_SIZE) + f'/{str(config.ADAPTIVE_RUN)}/lu_am_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'rb') as f:
            lu_am_object = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/temp_data/' + str(config.WINDOW_SIZE) + f'/{str(config.ADAPTIVE_RUN)}/seg_c_{str(config.WINDOW_SIZE)}_{str(config.ADAPTIVE_RUN)}.pkl', 'rb') as f:
            segment_container = pickle.load(f)
            lc_p_c = LCProbabilityContainer(segment_container, lc_am_object, lu_am_object, runtype="adaptive")
    elif runtype == "adaptive-full":
        with open(config.MODELS_DIRECTORY + '/new_data/' + str(config.WINDOW_SIZE) + f'/rgb_c_{str(config.WINDOW_SIZE)}.pkl', 'rb') as f:
            rgb_container = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/new_data/' + str(config.WINDOW_SIZE) + f'/lc_am_{str(config.WINDOW_SIZE)}.pkl', 'rb') as f:
            lc_am_object = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/new_data/' + str(config.WINDOW_SIZE) + f'/lu_am_{str(config.WINDOW_SIZE)}.pkl', 'rb') as f:
            lu_am_object = pickle.load(f)
        with open(config.MODELS_DIRECTORY + '/new_data/' + str(config.WINDOW_SIZE) + f'/seg_c_{str(config.WINDOW_SIZE)}.pkl', 'rb') as f:
            segment_container = pickle.load(f)
            lc_p_c = LCProbabilityContainer(segment_container, lc_am_object, lu_am_object, runtype="adaptive-full")
    return lc_p_c, lu_am_object, segment_container, lc_am_object, rgb_container


def compute_lu_annotation_array(k_fold, lu_am):
    lu_annotation_files = sorted(glob.glob(os.path.join(config.LU_ANNOTATION_DIRECTORY, '*.shp')))
    # For this k-fold, update the lc_am and lu_am training idxs
    for i, lu_ann_path in enumerate(lu_annotation_files):
        with fiona.open(lu_ann_path) as shapefile:
            lu_am.class_training_idxs[i] = [j for j, feature in enumerate(shapefile) if
                                            int(feature['properties']['k_fold_i']) != k_fold]
            lu_am.class_polygons[i] = [shape(feature['geometry']) for feature in shapefile]

    # And compute LU annotation array using the validation polygons for this k
    height, width = config.TIF.shape
    lu_annotation_array = -1 * np.ones(height * width, dtype=np.int8)
    for class_value in lu_am.class_polygons:
        for poly_idx in range(len(lu_am.class_polygons[class_value])):
            if poly_idx not in lu_am.class_training_idxs[class_value]:
                pixel_idxs = run.get_shape_pixel_idxs(lu_am.class_polygons[class_value][poly_idx])
                lu_annotation_array[pixel_idxs] = class_value
    return lu_annotation_array.reshape(config.TIF.shape)


def compute_classes_array_from_seg_c_p_c(seg_c_p_c_file_path, seg_c_object):
    return np.load(seg_c_p_c_file_path).argmax(axis=-1)[seg_c_object.segment_map].reshape(config.TIF.shape)


def compute_classes_array_from_pixelwise_output(lu_prob_path, pixel_map_path):
    pixel_map = np.load(pixel_map_path)
    lu_prob_array = np.load(lu_prob_path).argmax(axis=-1)
    lu_prob_image_dims = -1 * np.ones(config.TIF.shape)
    lu_prob_image_dims = lu_prob_image_dims.flatten()
    lu_prob_image_dims[pixel_map] = lu_prob_array
    return lu_prob_image_dims.reshape(config.TIF.shape)


def compute_correct_and_incorrect_amounts(classes_array, lu_annotation_array, lu_am):
    correct_pixels = np.zeros(config.NUMB_LU_CLASSES)
    incorrect_pixels = np.zeros(config.NUMB_LU_CLASSES)

    # For each LU class
    for class_index, class_value in enumerate(lu_am.class_polygons):
        # Compute how many of the LC Validation Annotations agree with the JDL output
        correct_pixels[class_index] = np.sum(
            (lu_annotation_array == classes_array) & (lu_annotation_array == class_value))
        # Compute how many of the LC Validation Annotations disagree with the JDL output
        incorrect_pixels[class_index] = np.sum(
            (lu_annotation_array != classes_array) & (lu_annotation_array == class_value))

    return correct_pixels, incorrect_pixels

# Update this to work for all folds - Same as two below
def compute_obia_svm_accuracies(numb_folds=5):
    _, lu_am, seg_c, _, _ = load_files()

    svm_correct = np.zeros((numb_folds, config.NUMB_LU_CLASSES))      
    svm_incorrect = np.zeros((numb_folds, config.NUMB_LU_CLASSES))  

    for fpath in glob.glob(os.path.join(config.MODELS_DIRECTORY, "*.npy")):

        k = int(fpath.split('.npy')[0][-1])

        classes_array = compute_classes_array_from_seg_c_p_c(fpath, seg_c)
        lu_annotation_array = compute_lu_annotation_array(k, lu_am)

        #oa, per_class_oa = compute_lu_accuracies_from_seg_c_p_c(classes_array, lu_annotation_array, lu_am) # Currently un-needed but kept for sanity

        correct, incorrect = compute_correct_and_incorrect_amounts(classes_array, lu_annotation_array, lu_am)
        svm_correct[k] = correct
        svm_incorrect[k] = incorrect

    averaged_accuracy = (svm_correct.sum(axis=1) / (svm_correct.sum(axis=1) + svm_incorrect.sum(axis=1))).mean()
    average_per_class_accuracy = (svm_correct / (svm_correct + svm_incorrect)).mean(axis=0)

    with open(os.path.join(config.MODELS_DIRECTORY, "overall_accuracy_over_" + str(numb_folds) + "_folds.json"), "w") as f:
        json.dump({"overall_accuracy": averaged_accuracy}, f)
    with open(os.path.join(config.MODELS_DIRECTORY, "per_class_accuracy_over_" + str(numb_folds) + "_folds.json"), "w") as f:
        json.dump(
            {"class" + str(class_value): average_per_class_accuracy[class_value] for class_value in range(len(average_per_class_accuracy))}, f)

def compute_pixelwise_accuracies(numb_folds=5):
    _, lu_am, seg_c, _, _ = load_files()
    dirpaths = [dirpath for dirpath in glob.glob(config.MODELS_DIRECTORY) if os.path.isdir(dirpath)]
    for dirpath in dirpaths:
        print("\n####################################")
        print(f"ANALYSING DATA IN FOLDER {dirpath}")
        print("####################################\n")
        
        pixelwise_correct = np.zeros((numb_folds, config.NUMB_LU_CLASSES))
        pixelwise_incorrect = np.zeros((numb_folds, config.NUMB_LU_CLASSES))

        for k in range(numb_folds):
            fpath_map = dirpath + "/" + "pixel_idx_map_" + str(k) + ".npy"
            fpath_prob = dirpath + "/" + "pixel_probabilities_pixelwise_" + str(k) + ".npy"
            classes_array = compute_classes_array_from_pixelwise_output(fpath_prob, fpath_map)
            lu_annotation_array = compute_lu_annotation_array(k, lu_am)
            correct, incorrect = compute_correct_and_incorrect_amounts(classes_array, lu_annotation_array, lu_am)

            pixelwise_correct[k] = correct
            pixelwise_incorrect[k] = incorrect

        averaged_accuracy = (pixelwise_correct.sum(axis=1) / (pixelwise_correct.sum(axis=1) + pixelwise_incorrect.sum(axis=1))).mean()
        average_per_class_accuracy = (pixelwise_correct / (pixelwise_correct + pixelwise_incorrect)).mean(axis=0)
            
        

        with open(os.path.join(dirpath, "oa_averaged_over_k_folds.json"), "w") as f:
            json.dump({"overall_accuracy": averaged_accuracy}, f)
        with open(os.path.join(dirpath, "per_class_accuracy_average.json"), "w") as f:
            json.dump(
                {"class" + str(class_value): average_per_class_accuracy[class_value] for class_value in
                 range(len(average_per_class_accuracy))},
                f)


def compute_multiscale_accuracies(numb_folds=5, runtype=None):
    if runtype == None:
        _, lu_am, seg_c, _, _ = load_files()
    if runtype == "adaptive":
        _, lu_am, seg_c, _, _ = load_files(runtype="adaptive")
    if runtype == "adaptive-full":
        _, lu_am, seg_c, _, _ = load_files(runtype="adaptive-full")
    
    # dirpaths = [dirpath for dirpath in glob.glob("models/multi_scale/96/*") if os.path.isdir(dirpath)]
    # for dirpath in dirpaths:

    length = len("ANALYSING DATA IN FOLDER ") + len(config.MODELS_DIRECTORY) + 1
    print("\n" + length * '#')
    print(f"ANALYSING DATA IN FOLDER {config.MODELS_DIRECTORY}")
    print(length * '#' + "\n")

    ms_correct = np.zeros((numb_folds, config.NUMB_LU_CLASSES))
    ms_incorrect = np.zeros((numb_folds, config.NUMB_LU_CLASSES))

    if runtype == "adaptive":
        path = os.path.join(config.MODELS_DIRECTORY, 'temp_data', str(config.WINDOW_SIZE), str(config.ADAPTIVE_RUN))
    elif runtype == "adaptive-full":
        path = os.path.join(config.MODELS_DIRECTORY, 'new_data', str(config.WINDOW_SIZE))
    else:
        path = os.path.join(config.MODELS_DIRECTORY)

    for fpath in glob.glob(path + "/*.npy"):

        print(f"\nANALYSING FILE: {fpath}")

        k = int(fpath.split('.npy')[0][-1])

        classes_array = compute_classes_array_from_seg_c_p_c(fpath, seg_c)
        lu_annotation_array = compute_lu_annotation_array(k, lu_am)

        correct, incorrect = compute_correct_and_incorrect_amounts(classes_array, lu_annotation_array, lu_am)
        ms_correct[k] = correct
        ms_incorrect[k] = incorrect

    averaged_accuracy = (ms_correct.sum(axis=1) / (ms_correct.sum(axis=1) + ms_incorrect.sum(axis=1))).mean()
    average_per_class_accuracy = (ms_correct / (ms_correct + ms_incorrect)).mean(axis=0)

    with open(os.path.join(os.path.split(fpath)[0], f"oa_averaged_over_{str(numb_folds)}_folds.json"), "w") as f:
        json.dump({"overall_accuracy": averaged_accuracy}, f)
    with open(os.path.join(os.path.split(fpath)[0], "per_class_accuracy_average.json"), "w") as f:
        json.dump(
            {"class" + str(class_value): average_per_class_accuracy[class_value] for class_value in
                range(len(average_per_class_accuracy))},
            f)

def compute_benchmark_ocnn_accuracies(numb_folds=5):
    _, lu_am, seg_c, _, _ = load_files()

    length = len("ANALYSING DATA IN FOLDER ") + len(config.MODELS_DIRECTORY) + 1
    print("\n" + length * '#')
    print(f"ANALYSING DATA IN FOLDER {config.MODELS_DIRECTORY}")
    print(length * '#' + "\n")

    benchmark_correct = np.zeros((numb_folds, config.NUMB_LU_CLASSES))
    benchmark_incorrect = np.zeros((numb_folds, config.NUMB_LU_CLASSES))

    for fpath in glob.glob(os.path.join(config.MODELS_DIRECTORY) + "/*.npy"):

        print(f"\nANALYSING FILE: {fpath}")

        k = int(fpath.split('.npy')[0][-1])

        classes_array = compute_classes_array_from_seg_c_p_c(fpath, seg_c)
        lu_annotation_array = compute_lu_annotation_array(k, lu_am)

        correct, incorrect = compute_correct_and_incorrect_amounts(classes_array, lu_annotation_array, lu_am)

        benchmark_correct[k] = correct
        benchmark_incorrect[k] = incorrect


    averaged_accuracy = (benchmark_correct.sum(axis=1) / (
                benchmark_correct.sum(axis=1) + benchmark_incorrect.sum(axis=1))).mean()
    average_per_class_accuracy = (benchmark_correct / (benchmark_correct + benchmark_incorrect)).mean(axis=0)

    with open(os.path.join(os.path.split(fpath)[0], f"benchmark_oa_averaged_over_{str(numb_folds)}_folds.json"), "w") as f:
        json.dump({"overall_accuracy": averaged_accuracy}, f)
    with open(os.path.join(os.path.split(fpath)[0], "benchmark_per_class_accuracy_average.json"), "w") as f:
        json.dump(
            {"class" + str(class_value): average_per_class_accuracy[class_value] for class_value in
                range(len(average_per_class_accuracy))},
            f)



def compute_adaptive_multiscale_accuracies(numb_folds=5, runtype=None):
    if runtype == None:
        _, lu_am, seg_c, _, _ = load_files()
    if runtype == "adaptive":
        _, lu_am, seg_c, _, _ = load_files(runtype="adaptive")
    if runtype == "adaptive-full" or runtype == "overall-acc":
        _, lu_am, seg_c, _, _ = load_files(runtype="adaptive-full")
    
    # dirpaths = [dirpath for dirpath in glob.glob("models/multi_scale/96/*") if os.path.isdir(dirpath)]
    # for dirpath in dirpaths:

    length = len("ANALYSING DATA IN FOLDER ") + len(config.MODELS_DIRECTORY) + 1
    print("\n" + length * '#')
    print(f"ANALYSING DATA IN FOLDER {config.MODELS_DIRECTORY}")
    print(length * '#' + "\n")

    ms_correct = np.zeros((numb_folds, config.NUMB_LU_CLASSES))
    ms_incorrect = np.zeros((numb_folds, config.NUMB_LU_CLASSES))

    if runtype == "adaptive":
        path = os.path.join(config.MODELS_DIRECTORY, 'temp_data', str(config.WINDOW_SIZE), str(config.ADAPTIVE_RUN))
    elif runtype == "adaptive-full" or runtype == "overall-acc":
        path = os.path.join(config.MODELS_DIRECTORY, 'new_data', str(config.WINDOW_SIZE))
    else:
        path = os.path.join(config.MODELS_DIRECTORY)

    for fpath in glob.glob(path + "/*.npy"):

        print(f"\nANALYSING FILE: {fpath}")

        k = int(fpath.split('.npy')[0][-1])

        classes_array = compute_classes_array_from_seg_c_p_c(fpath, seg_c)
        lu_annotation_array = compute_lu_annotation_array(k, lu_am)

        correct, incorrect = compute_correct_and_incorrect_amounts(classes_array, lu_annotation_array, lu_am)
        ms_correct[k] = correct
        ms_incorrect[k] = incorrect

    if runtype == "overall-acc":
        return ms_correct, ms_incorrect
    else:

        averaged_accuracy = (ms_correct.sum(axis=1) / (ms_correct.sum(axis=1) + ms_incorrect.sum(axis=1))).mean()
        average_per_class_accuracy = (ms_correct / (ms_correct + ms_incorrect)).mean(axis=0)

        with open(os.path.join(os.path.split(fpath)[0], f"oa_averaged_over_{str(numb_folds)}_folds.json"), "w") as f:
            json.dump({"overall_accuracy": averaged_accuracy}, f)
        with open(os.path.join(os.path.split(fpath)[0], "per_class_accuracy_average.json"), "w") as f:
            json.dump(
                {"class" + str(class_value): average_per_class_accuracy[class_value] for class_value in
                    range(len(average_per_class_accuracy))},
                f)


if __name__ == '__main__':
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "DIN Alternate"

    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--models_directory", help="path to models directory e.g. models/JDLs/31032019_0")
    parser.add_argument("-tp", "--tif_path", help="path to data .tif",
                        default="geo_data/K_R3C4/NI_Kano_20Q1_V0_R3C4.tif")
    parser.add_argument("-lc", "--lc_directory",
                        help="path to directory containing lc annotations e.g. geo_data/L_R6C6/mlp_labels1",
                        default='geo_data/K_R3C4/lc_annotations')
    parser.add_argument("-lu", "--lu_directory",
                        help="path to directory containing lu annotations e.g. geo_data/L_R6C6/ocnn_labels1",
                        default='geo_data/K_R3C4/lu_annotations')
    parser.add_argument("-s", "--segmentation_file",
                        help="path to segmentation shapefile",
                        default='geo_data/K_R3C4/R3C430_8_196_full_tif.shp')
    parser.add_argument("-ws", "--window_size", help="number of pixels used as window height/width", default=96)
    args = parser.parse_args()

    """ PATH SPECIFICATION VALIDATION """
    if not os.path.isdir(args.models_directory):
        print("Specified model directory does not exist, creating")
        os.mkdir(args.models_directory)
    if not os.path.isfile(args.tif_path):
        print("Cannot find specified .tif file, please check and restart")
        quit()
    if not os.path.isdir(args.lc_directory):
        print("Cannot find specified LC annotation directory, please check and restart")
        quit()
    if not os.path.isdir(args.lu_directory):
        print("Cannot find specified LU annotation directory, please check and restart")
        quit()
    if not os.path.isfile(args.segmentation_file):
        print("Cannot find specified segmentation shapefile, please check and restart")
        quit()

    config.TIF_PATH = args.tif_path
    config.MODELS_DIRECTORY = args.models_directory
    config.LC_ANNOTATION_DIRECTORY = args.lc_directory
    config.LU_ANNOTATION_DIRECTORY = args.lu_directory
    config.SEGMENTATION_SHAPEFILE = args.segmentation_file
    config.NUMB_LC_CLASSES = len(glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp'))
    config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))
    config.TIF = rasterio.open(config.TIF_PATH)
    config.WINDOW_SIZE = int(args.window_size)

    run.config.TIF_PATH = args.tif_path
    run.config.MODELS_DIRECTORY = args.models_directory
    run.config.LC_ANNOTATION_DIRECTORY = args.lc_directory
    run.config.LU_ANNOTATION_DIRECTORY = args.lu_directory
    run.config.SEGMENTATION_SHAPEFILE = args.segmentation_file
    run.config.NUMB_LC_CLASSES = len(glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp'))
    run.config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))
    run.config.TIF = rasterio.open(config.TIF_PATH)
    run.config.WINDOW_SIZE = int(args.window_size)
    run.config.IDXS_TIF = run.make_idxs_tif()

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

    def choice_input():
        print("\n##############################################")
        print("PLEASE CHOSE A MODEL TYPE TO COMPUTE ACCURACIES")
        print("################################################\n")
        print("1. Multi-Scale CNN")
        print("2. Benchmark OCNN")
        print("3. OBIA SVM")
        print("4. Pixelwise CNN\n")
        choice = input("Please enter one of the numbers above as an option (enter q to quit): ")
        print("")

        if choice in ("1", "2", "3", "4"):
            choice = int(choice)
            if choice == 1:
                compute_multiscale_accuracies()
            elif choice == 2:
                compute_benchmark_ocnn_accuracies(numb_folds=5)
            elif choice == 3:
                compute_obia_svm_accuracies(numb_folds=5)
            elif choice == 4:
                compute_pixelwise_accuracies(numb_folds=1)
        elif choice == "q":
            quit()
        else:
            print("That is not a valid option, please enter a valid option")
            choice_input()

    choice_input()
