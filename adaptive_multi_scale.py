import argparse
from math import comb
import os
from run import make_idxs_tif
import config
import rasterio
import glob
import tensorflow as tf
from time import time
import run
import pickle
import project_metrics
import json
from shutil import copyfile, copy
from collections import Counter
import operator
import numpy as np

def create_initial_data(window_size, run_fold):
    print("\n########################################")
    print(f"CREATING DATA FOR WINDOW SIZE {window_size}, RUN {run_fold}")
    print("########################################")

    config.WINDOW_SIZE = window_size
    config.ADAPTIVE_RUN = run_fold

    window_size_path = str(window_size)
    folder = str(run_fold)
    
    if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'temp_data')):
        os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'temp_data'))

    if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'temp_data', window_size_path)):
            print("[NOTICE] Window size directory not found, creating directory...")
            os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'temp_data', window_size_path))

    if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'temp_data', window_size_path, folder)):
        print("[NOTICE] Initial Data Run directory not found, creating directory...")
        os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'temp_data', window_size_path, folder))

    print("\n###########################################")
    print("LOADING ANNOTATION MANAGERS AND CONTAINERS")
    print("###########################################\n")

    start = time()

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'lc_am_{window_size_path}_{folder}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'lc_am_{window_size_path}_{folder}.pkl'), 'rb') as f:
            lc_am = pickle.load(f)
    else:
        print("\n###############################################")
        print("LAND COVER ANNOTATION MANAGER NOT FOUND - CREATING")
        print("#################################################\n")
        lc_am = run.LandCoverAnnotationManager(runtype="adaptive")

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'lu_am_{window_size_path}_{folder}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'lu_am_{window_size_path}_{folder}.pkl'), 'rb') as f:
            lu_am = pickle.load(f)
    else:
        print("\n###############################################")
        print("LAND USE ANNOTATION MANAGER NOT FOUND - CREATING")
        print("#################################################\n")
        lu_am = run.LandUseAnnotationManager(runtype="adaptive")

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'rgb_c_{window_size_path}_{folder}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'rgb_c_{window_size_path}_{folder}.pkl'), 'rb') as f:
            rgb_container = pickle.load(f)
    else:
        print("\n#################################")
        print("RGB CONTAINER NOT FOUND - CREATING")
        print("###################################\n")
        rgb_container = run.RGBContainer(runtype="adaptive")

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'seg_c_{window_size_path}_{folder}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'temp_data',  window_size_path, folder, f'seg_c_{window_size_path}_{folder}.pkl'), 'rb') as f:
            segment_container = pickle.load(f)
    else:
        print("\n#####################################")
        print("SEGMENT CONTAINER NOT FOUND - CREATING")
        print("#######################################\n")
        segment_container = run.SegmentContainer(runtype="adaptive")

    config.NUMB_SEGMENTS = segment_container.prob_container.probabilities.shape[0]
    lc_p_container = run.LCProbabilityContainer(segment_container, lc_am, lu_am, runtype="adaptive")
    
    end = time()

    print('Time taken (initiation): %s' % (end - start))
    print("Window size: %s" % str(config.WINDOW_SIZE))

    run.lc_am = lc_am
    run.lu_am = lu_am
    run.rgb_container = rgb_container
    run.segment_container = segment_container
    run.lc_p_container = lc_p_container

def initial_fold_accuracy_generation(window_size, run):
    print("\n########################################")
    print(f"GENERATING ACCURACIES FOR INITIAL FOLD")
    print("#########################################")

    project_metrics.config.WINDOW_SIZE = window_size
    project_metrics.config.ADAPTIVE_RUN = run
    project_metrics.compute_adaptive_multiscale_accuracies(numb_folds=3, runtype="adaptive")

def shapefile_assignment(window_sizes):

    print("\n#######################################")
    print(f"ASSIGNING CLASSES TO NEW WINDOW SIZES")
    print("#######################################")

    master_dict = {}
    organised_dict = {}

    print("\n#############################################")
    print(f"RETRIEVING CLASS ACCURACIES FOR ALL WINDOWS")
    print("#############################################")

    for class_idx in range(config.NUMB_LU_CLASSES):
        # class idx = 0 - 6 through loop
        class_dict = {}
        for window_size in window_sizes:
            for i in range(4):
                path = os.path.join(config.MODELS_DIRECTORY, 'temp_data', str(window_size), str(i), 'per_class_accuracy_average.json')
                with open(path, "r") as f:
                    data = json.load(f)
                class_dict[f"{window_size}-{i}"] = data[f"class{class_idx}"]
        master_dict[class_idx] = class_dict


    print("\n#######################################")
    print(f"DETERMINING BEST WINDOW SIZE FOR DATA")
    print("#######################################")

    for class_idx, accuracies in master_dict.items():
        #print(class_idx, accuracies)
        votes = {}
        for i in range(len(accuracies.items()) // len(window_sizes)):
            accs = {}
            for window_size in window_sizes:
                accs[f"{window_size}-{i}"] = accuracies[f"{window_size}-{i}"]

            max_val = max(accs.items(), key=operator.itemgetter(1))[0]
            votes[i] = max_val.split("-")[0]

        organised_dict[class_idx] = Counter(votes.values()).most_common(1)[0][0]
    print("CLASSES ASSIGNED TO THE FOLLOWING WINDOWS")
    print(organised_dict)

    print("\n################################")
    print(f"ASSIGNING NEW DATA DIRECTORIES")
    print("################################")

    if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles')):
        os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles'))

    for window_size in window_sizes:
        if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size))):
            os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size)))


    shapefiles = sorted(glob.glob(config.LU_ANNOTATION_DIRECTORY + "/*.shp"))

    for class_idx, window_size in organised_dict.items():
        shapefile = shapefiles[int(class_idx)]
        name = shapefile.split("\\")[-1].split(".")[0]
        for name in glob.glob(config.LU_ANNOTATION_DIRECTORY + f'/{name}.*'):
            copy(name, os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size)))

    new_windows = []

    for window_size in window_sizes:
        file_num = len(os.listdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size))))
        if file_num == 0:
            os.rmdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size)))
        elif file_num != 0:
            new_windows.append(window_size)

        
    return new_windows

def create_new_data(window_size):
    print("\n#####################################")
    print(f"CREATING NEW DATA FOR WINDOW SIZE {window_size}")
    print("#####################################")

    config.WINDOW_SIZE = window_size
    window_size_path = str(window_size)
    
    if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'new_data')):
        os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'new_data'))

    if not os.path.isdir(os.path.join(config.MODELS_DIRECTORY, 'new_data', window_size_path)):
            print("[NOTICE] Window size directory not found, creating directory...")
            os.mkdir(os.path.join(config.MODELS_DIRECTORY, 'new_data', window_size_path))

    print("\n###########################################")
    print("LOADING ANNOTATION MANAGERS AND CONTAINERS")
    print("###########################################\n")

    start = time()

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'lc_am_{window_size_path}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'lc_am_{window_size_path}.pkl'), 'rb') as f:
            lc_am = pickle.load(f)
    else:
        print("\n###############################################")
        print("LAND COVER ANNOTATION MANAGER NOT FOUND - CREATING")
        print("#################################################\n")
        lc_am = run.LandCoverAnnotationManager(runtype="adaptive-full")

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'lu_am_{window_size_path}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'lu_am_{window_size_path}.pkl'), 'rb') as f:
            lu_am = pickle.load(f)
    else:
        print("\n###############################################")
        print("LAND USE ANNOTATION MANAGER NOT FOUND - CREATING")
        print("#################################################\n")
        lu_am = run.LandUseAnnotationManager(runtype="adaptive-full")

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'rgb_c_{window_size_path}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'rgb_c_{window_size_path}.pkl'), 'rb') as f:
            rgb_container = pickle.load(f)
    else:
        print("\n#################################")
        print("RGB CONTAINER NOT FOUND - CREATING")
        print("###################################\n")
        rgb_container = run.RGBContainer(runtype="adaptive-full")

    if os.path.isfile(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'seg_c_{window_size_path}.pkl')):
        with open(os.path.join(config.MODELS_DIRECTORY, 'new_data',  window_size_path, f'seg_c_{window_size_path}.pkl'), 'rb') as f:
            segment_container = pickle.load(f)
    else:
        print("\n#####################################")
        print("SEGMENT CONTAINER NOT FOUND - CREATING")
        print("#######################################\n")
        segment_container = run.SegmentContainer(runtype="adaptive-full")

    config.NUMB_SEGMENTS = segment_container.prob_container.probabilities.shape[0]
    lc_p_container = run.LCProbabilityContainer(segment_container, lc_am, lu_am, runtype="adaptive-full")
    
    end = time()

    print('Time taken (initiation): %s' % (end - start))
    print("Window size: %s" % str(config.WINDOW_SIZE))

    run.lc_am = lc_am
    run.lu_am = lu_am
    run.rgb_container = rgb_container
    run.segment_container = segment_container
    run.lc_p_container = lc_p_container
            
def full_5_fold_accuracy_generation(window_size):
    print("\n##########################################")
    print(f"GENERATING ACCURACIES FOR NEW DATA 5-FOLD")
    print("###########################################")

    project_metrics.config.WINDOW_SIZE = window_size
    project_metrics.config.LU_ANNOTATION_DIRECTORY = os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size))
    project_metrics.config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))
    project_metrics.compute_adaptive_multiscale_accuracies(numb_folds=5, runtype="adaptive-full")

def overall_accuracy_combiner():
    shapefile_assignments = {}
    shapefile_order_alph = []

    for shp_dir in os.listdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles')):
        list = [item.split(".")[0] for item in os.listdir(os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', shp_dir)) if item.split(".")[1] == "shp"]
        for i in range(len(list)):
            shapefile_assignments[list[i]] = shp_dir
            shapefile_order_alph.append(list[i])
    sorted_shapefiles = sorted(shapefile_assignments.items())
    alph_shapefile = sorted(shapefile_order_alph)

    list = os.listdir(os.path.join(config.MODELS_DIRECTORY, 'new_data'))
    combined_accuracies = {}

    for dir in range(len(list)):
        with open(os.path.join(config.MODELS_DIRECTORY, 'new_data', list[dir], 'per_class_accuracy_average.json')) as f:
            dir_data = json.load(f)

        dir_shp = [k for k,v in shapefile_assignments.items() if v == list[dir]]
        
        for i in range(len(sorted_shapefiles)):
            if sorted_shapefiles[i][1] == list[dir]:
                main_position = alph_shapefile.index(sorted_shapefiles[i][0])
                data_position = dir_shp.index(sorted_shapefiles[i][0])
                combined_accuracies[f"class{main_position}"] = dir_data[f"class{data_position}"]

    final_acc = {}
    for item in sorted(combined_accuracies.items()):
        final_acc[item[0]] = item[1]

    with open(os.path.join(config.MODELS_DIRECTORY, "combined_accuracies_over_5_folds_all_ws.json"), "w") as f:
            json.dump(
                {str(class_value): acc for class_value, acc in
                 final_acc.items()},
                f)

def overall_accuracy_generator(window_sizes):

    window_sizes = window_sizes[0]


    correct = []
    incorrect = []

    for window_size in window_sizes:
        project_metrics.config.WINDOW_SIZE = window_size
        project_metrics.config.LU_ANNOTATION_DIRECTORY = os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size))
        project_metrics.config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))
        amuse_corr, amuse_incorr = project_metrics.compute_adaptive_multiscale_accuracies(numb_folds=5, runtype="overall-acc")
        correct.append(amuse_corr)
        incorrect.append(amuse_incorr)

    correct_total = 0
    incorrect_total = 0

    for item in correct:
        sum = item.sum(axis=1)
        correct_total += sum

    for item in incorrect:
        sum = item.sum(axis=1)
        incorrect_total += sum

    averaged_accuracy = (correct_total / (correct_total + incorrect_total)).mean()

    with open(os.path.join(config.MODELS_DIRECTORY, f"oa_combined_averaged_over_5_folds.json"), "w") as f:
            json.dump({"overall_accuracy": averaged_accuracy}, f)



def main_run(window_sizes):
    print("\n######################################################################")
    print(f"RUNNING ADAPTIVE MS-CNN FOR WINDOW SIZES: {window_sizes}")
    print("########################################################################")

    # for window_size in window_sizes:
    #     for i in range(4):
    #         lc_am = None
    #         lu_am = None
    #         rgb_container = None
    #         segment_container = None
    #         lc_p_container = None

    #         create_initial_data(window_size, i)
    #         run.adaptive_multiscale_module(3, runtype=f"adaptive-{i}")
    #         initial_fold_accuracy_generation(window_size, i)

    window_sizes = shapefile_assignment(window_sizes)

    # for window_size in window_sizes:
    #     lc_am = None
    #     lu_am = None
    #     rgb_container = None
    #     segment_container = None
    #     lc_p_container = None

    #     config.LU_ANNOTATION_DIRECTORY = os.path.join(config.MODELS_DIRECTORY, 'temp_shapefiles', str(window_size))
    #     config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))

    #     create_new_data(window_size)

    #     run.adaptive_multiscale_module(5, runtype="adaptive-full")
    #     full_5_fold_accuracy_generation(window_size)
    # overall_accuracy_combiner()
    overall_accuracy_generator([window_sizes])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--models_directory", help="path to models directory e.g. models/adaptive_multi_scale/experiment_1")
    parser.add_argument("-tp", "--tif_path", help="path to data .tif file",
    default="geo_data/K_R3C4/NI_Kano_20Q1_V0_R3C4.tif")
    parser.add_argument("-lc", "--lc_annotations", help="path to directory containing LC annotations e.g. geo_data/K_R3C4/lc_annotations", 
        default="geo_data/K_R3C4/lc_annotations")
    parser.add_argument("-lu", "--lu_annotations", help="path to directory containing LC annotations e.g. geo_data/K_R3C4/lu_annotations",
        default="geo_data/K_R3C4/lu_annotations")
    parser.add_argument("-s", "--segmentation_file", help="path to segmentation shapefile e.g. geo_data/K_R3C4/R3C430_8_196_full_tif.shp", 
        default="geo_data/K_R3C4/R3C430_8_196_full_tif.shp")
    parser.add_argument("-ws", "--window_sizes", 
    help="List of window sizes to be used in training of adaptive model, must be defined as WS WS WS e.g. 48 96 120 160", nargs='+', type=int,
            default=[48,96])
    args = parser.parse_args()

    """ PATH SPECIFICATION VALIDATION """
    if not os.path.isdir(args.models_directory):
        print("[NOTICE] Specified model directory does not exist! \n Creating directory...")
        os.mkdir(args.models_directory)
    if not os.path.isfile(args.tif_path):
        print("[FATAL ERROR] Cannot find specified .tif file, please check and restart")
        quit()
    if not os.path.isdir(args.lc_annotations):
        print("[FATAL ERROR] Cannot find specified LC annotation directory, please check and restart")
        quit()
    if not os.path.isdir(args.lu_annotations):
        print("[FATAL ERROR] Cannot find specified LU annotation directory, please check and restart")
    if not os.path.isfile(args.segmentation_file):
        print("[FATAL ERROR] Cannot find specified segmentation shapefile, please check and restart")   

    config.TIF_PATH = args.tif_path
    config.TIF = rasterio.open(config.TIF_PATH)
    config.IDXS_TIF = make_idxs_tif()
    config.MODELS_DIRECTORY = args.models_directory
    config.LC_ANNOTATION_DIRECTORY = args.lc_annotations
    config.LU_ANNOTATION_DIRECTORY = args.lu_annotations
    config.SEGMENTATION_SHAPEFILE = args.segmentation_file
    config.NUMB_LC_CLASSES = len(glob.glob(config.LC_ANNOTATION_DIRECTORY + '/*.shp'))
    config.NUMB_LU_CLASSES = len(glob.glob(config.LU_ANNOTATION_DIRECTORY + '/*.shp'))

    window_sizes = args.window_sizes

    print("\n##########################")
    print("CONFIRMING GPUS AVAILABLE")
    print("##########################")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialised
            print(f"[FATAL ERROR] Error occured when attempting to load GPUS \
                details of this error are shown below \n\n {e}")


    main_run(window_sizes)


    





    
    

    