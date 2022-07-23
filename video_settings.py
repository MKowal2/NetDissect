######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                            # turning on the testmode means the code will run on a small dataset.
CLEAN = True                                # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'resnet18'                          # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'imagenet'                        # model trained on: places365 or imagenet
IMAGE_DATASETS = ['places365', 'imagenet']
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
FO_AHEAD = 1                                # number of data items to prefetch ahead
SINGLE_THREAD = True                        # use a single process in dataloader
CATAGORIES = ["object", "part","scene","texture","color","material"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
# CATAGORIES = ["dynamic", "appearance","flow","color"] # concept categories that are chosen to detect: "dynamic", "appearance", "flow", "color"
# OUTPUT_FOLDER = "result/video_a2d_"+MODEL+"_"+DATASET # result will be stored in this folder
# OUTPUT_FOLDER = "result/video_a2d_"+MODEL+"_"+DATASET # result will be stored in this folder
OUTPUT_FOLDER = "result/PytorchLoader_"+MODEL+"_"+DATASET # result will be stored in this folder
VIDEO_INPUT = True                         # model takes as input videos instead of images
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
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

if MODEL != 'alexnet':
    # DATA_DIRECTORY = 'dataset/video_broden3_224'
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    # DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

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
elif MODEL == 'slowonly8x8':
    FEATURE_NAMES = ['layer4']


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
