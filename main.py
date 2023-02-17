import argparse
import os
import config
import rasterio
from run import *
import run
import glob
import tensorflow as tf
from time import time
import pickle
import adaptive_multi_scale

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
parser.add_argument("-ws", "--window_sizes", 
                    help="List of window sizes to be used in training of adaptive model, must be defined as WS WS WS e.g. 48 96 120 160", nargs='+', type=int,
                    default=[48,96])
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

run.config.TIF_PATH = args.tif_path
run.config.TIF = rasterio.open(config.TIF_PATH)
run.config.IDXS_TIF = run.make_idxs_tif()
run.config.MODELS_DIRECTORY = args.models_directory
run.config.LC_ANNOTATION_DIRECTORY = args.lc_annotations
run.config.LU_ANNOTATION_DIRECTORY = args.lu_annotations
run.config.SEGMENTATION_SHAPEFILE = args.segmentation_file
run.config.NUMB_LC_CLASSES = len(glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp'))
run.config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))


config.TIF_PATH = args.tif_path
config.TIF = rasterio.open(config.TIF_PATH)
config.IDXS_TIF = run.make_idxs_tif()
config.MODELS_DIRECTORY = args.models_directory
config.LC_ANNOTATION_DIRECTORY = args.lc_annotations
config.LU_ANNOTATION_DIRECTORY = args.lu_annotations
config.SEGMENTATION_SHAPEFILE = args.segmentation_file
config.NUMB_LC_CLASSES = len(glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp'))
config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))

if len(args.window_sizes) == 1:
    run.config.WINDOW_SIZE = int(args.window_sizes[0])
    config.WINDOW_SIZE = int(args.window_sizes[0])

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
        lc_am = run.LandCoverAnnotationManager()

    if os.path.isfile(config.MODELS_DIRECTORY + '/lu_am.pkl'):
        with open(config.MODELS_DIRECTORY + '/lu_am.pkl', 'rb') as f:
            lu_am = pickle.load(f)
    else:
        print("\n###############################################")
        print("LAND USE ANNOTATION MANAGER NOT FOUND - CREATING")
        print("#################################################\n")
        lu_am = run.LandUseAnnotationManager()

    if os.path.isfile(config.MODELS_DIRECTORY + '/rgb_c.pkl'):
        with open(config.MODELS_DIRECTORY + '/rgb_c.pkl', 'rb') as f:
            rgb_container = pickle.load(f)
    else:
        print("\n#################################")
        print("RGB CONTAINER NOT FOUND - CREATING")
        print("###################################\n")
        rgb_container = run.RGBContainer()

    if os.path.isfile(config.MODELS_DIRECTORY + '/seg_c.pkl'):
        with open(config.MODELS_DIRECTORY + '/seg_c.pkl', 'rb') as f:
            segment_container = pickle.load(f)
    else:
        print("\n#####################################")
        print("SEGMENT CONTAINER NOT FOUND - CREATING")
        print("#######################################\n")
        segment_container = run.SegmentContainer()

    config.NUMB_SEGMENTS = segment_container.prob_container.probabilities.shape[0]
    # Create LC Prob container using segment container, land class annotation 
    # manager and land use annotation manager
    lc_p_container = run.LCProbabilityContainer(segment_container, lc_am, lu_am)

    end = time()

    print('Time taken (initiation): %s' % (end - start))
    print("Window size: %s" % str(config.WINDOW_SIZE))

    run.lc_am = lc_am
    run.lu_am = lu_am
    run.rgb_container = rgb_container
    run.segment_container = segment_container
    run.lc_p_container = lc_p_container

def choice_input():
    print("\n###############################")
    print("PLEASE CHOSE A MODEL TYPE TO RUN")
    print("#################################\n")
    if len(args.window_sizes) == 1:
        print("1. Multi-Scale CNN")
        print("2. OCNN Benchmark")
        print("3. Pixelwise CNN")
        print("4. OBIA SVM")
        print("5. MS-CNN and OCNN Benchmark Dual Mode\n")
        choice = input("Please enter one of the numbers above as an option (enter q to quit): ")
        print("")
    else:
        print("[NOTE] You have provided multiple window sizes and therefore only one option is available")
        print("\n1. Adaptive Multi-Scale CNN\n")
        choice = input("Please enter one of the numbers above as an option (enter q to quit): ")
        print("")

    if choice in ("1", "2", "3", "4", "5"):
        choice = int(choice)
        if choice == 1:
            if len(args.window_sizes) == 1:
                fold = int(input("How many folds should be performed in Multi-Scale K-Fold Cross Validation (Integers > 0 Only)? "))
                fold = fold if fold > 0 else 1
                run.multiscale_module(fold)
            else:
                adaptive_multi_scale.main_run(args.window_sizes)
        elif choice == 2:
                fold = int(input("How many folds should be performed in OCNN K-Fold Cross Validation (Integers > 0 Only)? "))
                fold = fold if fold > 0 else 1
                run.ocnn_module(fold)
        elif choice == 3:
            numb_folds = int(input("How many folds should be performed in PixelWise K-Fold Cross Validation (Integers > 0 Only)? "))
            numb_folds = numb_folds if numb_folds > 0 else 1
            run.pixelwise_CNN(numb_folds)
        elif choice == 4:
            numb_folds = int(input("How many folds should be performed in OBIA-SVM K-Fold Cross Validation (Integers > 0 Only)? "))
            numb_folds = numb_folds if numb_folds > 0 else 1
            run.obia_svm(numb_folds)
        elif choice == 5:
            numb_folds = int(input("How many folds should be performed in dual MSCNN OCNN K-Fold Cross Validation (Integers > 0 Only)? "))
            numb_folds = numb_folds if numb_folds > 0 else 1
            run.dual_mscnn_ocnn(numb_folds)
    elif choice == "q":
        quit()
    else:
        print("That is not a valid option, please enter a valid option")
        choice_input()

auto_val = int(args.auto)

if(auto_val != 0 and isinstance(auto_val, int) and auto_val in (1, 2, 3)):
    choice = auto_val
    if choice == 1:
        if len(args.window_sizes) == 1:
            print("\n##############################")
            print("RUNNING AUTOMATIC 5-FOLD MS-CNN")
            print("################################\n")
            run.multiscale_module(5)
        else:
            print("\n##############################")
            print("RUNNING AUTOMATIC AMUSE")
            print("################################\n")
            adaptive_multi_scale.main_run(args.window_sizes)
    elif choice == 2:
        print("\n#####################################")
        print("RUNNING AUTOMATIC 5-FOLD Pixelwise-CNN")
        print("#######################################\n")
        run.pixelwise_CNN(5)
    elif choice == 3:
        print("\n################################")
        print("RUNNING AUTOMATIC 5-FOLD OBIA-SVM")
        print("##################################\n")
        run.obia_svm(5)
elif auto_val == 0:
    choice_input()
else:
    print("Error with automatic code parameter, please check")
    quit()
