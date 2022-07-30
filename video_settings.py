######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
SINGLE_THREAD = False                        # use a single process in dataloader
TEST_MODE = False                            # turning on the testmode means the code will run on a small dataset.
CLEAN = True                                # set to "True" if you want to clean the temporary large files after generating result
VIDEO_INPUT = True                           # model takes as input videos instead of images
MODEL = 'i3d'                                # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'kinetics'                         # model trained on:
# MODEL = 'resnet18'                          # model arch: resnet18, alexnet, resnet50, densenet161
# DATASET = 'imagenet'                        # model trained on:
IMAGE_DATASETS = ['places365', 'imagenet']
VIDEO_DATASETS = ['kinetics', 'ssv2']
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
FO_AHEAD = 1                                # number of data items to prefetch ahead
# CATEGORIES = ["object", "part","scene","texture","color","material"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
CATEGORIES = ["dynamic", "appearance","flow","color"] # concept categories that are chosen to detect: "dynamic", "appearance", "flow", "color"
# OUTPUT_FOLDER = "result/video_a2d_"+MODEL+"_"+DATASET # result will be stored in this folder
# OUTPUT_FOLDER = "result/video_a2d_"+MODEL+"_"+DATASET # result will be stored in this folder
OUTPUT_FOLDER = "result/AllLayers_Full_A2D_"+MODEL+"_"+DATASET # result will be stored in this folder
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
USE_OLD_LOADER = False
VIDEO_MEAN = False
RANDOM_INIT = True

if RANDOM_INIT:
    OUTPUT_FOLDER += "_RandWeight"

# MEAN = [109.5388, 118.6897, 124.6901] # previous mean

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if VIDEO_INPUT:
    DATA_DIRECTORY = 'dataset/A2D_video_broden4_224'
    IMG_SIZE = 224
else:
    if MODEL != 'alexnet':
        DATA_DIRECTORY = 'dataset/broden1_224'
        IMG_SIZE = 224
    else:
        # DATA_DIRECTORY = 'dataset/broden1_227'
        IMG_SIZE = 227


if VIDEO_INPUT:
    if DATASET == 'kinetics':
        NUM_CLASSES = 400
    elif DATASET == 'ssv2':
        NUM_CLASSES = 174
    if MODEL == 'slowonly8x8':
        FEATURE_NAMES = ['s1', 's2','s3','s4','s5']
        MIDDLE_FRAME = 4
    elif MODEL == 'i3d':
        FEATURE_NAMES = ['s1', 's2','s3','s4','s5']
        MIDDLE_FRAME = 2
    elif MODEL == 'c2d':
        FEATURE_NAMES = ['s5']
        MIDDLE_FRAME = 2
else:
    if DATASET == 'places365':
        NUM_CLASSES = 365
    elif DATASET == 'imagenet':
        NUM_CLASSES = 1000
    if MODEL == 'resnet18':
        FEATURE_NAMES = ['layer4']
        if DATASET == 'places365':
            MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
            MODEL_PARALLEL = True
        elif DATASET == 'imagenet':
            MODEL_FILE = None
            MODEL_PARALLEL = False
    elif MODEL == 'densenet161':
        FEATURE_NAMES = ['features']
        if DATASET == 'places365':
            MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
            MODEL_PARALLEL = False
    elif MODEL == 'resnet50':
        FEATURE_NAMES = ['layer4']
        if DATASET == 'places365':
            MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
            MODEL_PARALLEL = False


if VIDEO_INPUT:
    if TEST_MODE:
        WORKERS = 16
        BATCH_SIZE = 16
        TALLY_BATCH_SIZE = 16
        TALLY_AHEAD = 4
        INDEX_FILE = 'index_sm.csv'
        OUTPUT_FOLDER += "_test"
    else:
        WORKERS = 16
        BATCH_SIZE = 64
        TALLY_BATCH_SIZE = 24
        TALLY_AHEAD = 8
        INDEX_FILE = 'index.csv'
else:
    if TEST_MODE:
        WORKERS = 1
        BATCH_SIZE = 4
        TALLY_BATCH_SIZE = 1
        TALLY_AHEAD = 1
        INDEX_FILE = 'index.csv'
        OUTPUT_FOLDER += "_test"
    else:
        WORKERS = 16
        BATCH_SIZE = 1024
        TALLY_BATCH_SIZE = 24
        TALLY_AHEAD = 8
        INDEX_FILE = 'index.csv'
