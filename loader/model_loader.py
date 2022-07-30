import video_settings as settings
import torch
import torchvision


def loadmodel(hook_fn):
    if not settings.VIDEO_INPUT:
        if settings.MODEL_FILE is None:
            model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
        else:
            checkpoint = torch.load(settings.MODEL_FILE)
            if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                if settings.MODEL_PARALLEL:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                        'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
    else:
        model = load_video_model(settings)
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model



def load_video_model(settings):
    from loader.models.slowfast.slowfast.models.build import build_model
    from loader.models.slowfast.slowfast.utils.parser import load_config, parse_args
    from loader.models.slowfast.slowfast.config.defaults import assert_and_infer_cfg
    import loader.models.slowfast.slowfast.utils.checkpoint as cu
    if settings.MODEL == 'slowonly8x8':
        model_args = parse_args()
        model_args.cfg_files = ['loader/models/slowfast/configs/Kinetics/c2/SLOW_8x8_R50.yaml']
        if settings.DATASET == 'kinetics':
            cfg = load_config(model_args, 'loader/models/slowfast/configs/Kinetics/c2/SLOW_8x8_R50.yaml')
            cfg.TRAIN.CHECKPOINT_FILE_PATH = 'zoo/SLOWONLY_8x8_R50.pkl'
        elif settings.DATASET == 'ssv2':
            cfg = load_config(model_args, 'loader/models/slowfast/configs/Kinetics/c2/SLOW_8x8_R50.yaml')
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_FILE_PATH = 'zoo/SLOWONLY_8x8_ours_epoch_00030.pyth'
            cfg.MODEL.NUM_CLASSES = settings.NUM_CLASSES
        cfg = assert_and_infer_cfg(cfg)
        model = build_model(cfg)
    elif settings.MODEL == 'c2d':
        model_args = parse_args()
        model_args.cfg_files = ['loader/models/slowfast/configs/Kinetics/C2D_8x8_R50_IN1K.yaml']
        if settings.DATASET == 'kinetics':
            cfg = load_config(model_args, 'loader/models/slowfast/configs/Kinetics/C2D_8x8_R50_IN1K.yaml')
            cfg.TRAIN.CHECKPOINT_FILE_PATH = 'zoo/c2d_baseline_8x8_IN_pretrain_400k.pkl'
            cfg.TRAIN.CHECKPOINT_TYPE = "caffe2"
        cfg = assert_and_infer_cfg(cfg)
        model = build_model(cfg)
    elif settings.MODEL == 'i3d':
        model_args = parse_args()
        model_args.cfg_files = ['loader/models/slowfast/configs/Kinetics/c2/I3D_8x8_R50.yaml']
        if settings.DATASET == 'kinetics':
            cfg = load_config(model_args, 'loader/models/slowfast/configs/Kinetics/c2/I3D_8x8_R50.yaml')
            cfg.TRAIN.CHECKPOINT_FILE_PATH = 'zoo/I3D_8x8_R50.pkl'
        cfg = assert_and_infer_cfg(cfg)
        model = build_model(cfg)
    else:
        print('Model selected not available')
        raise NotImplementedError

    if not settings.RANDOM_INIT:
        cu.load_test_checkpoint(cfg, model)
    settings.SAMPLING_RATE = cfg.DATA.SAMPLING_RATE
    settings.NUM_FRAMES = cfg.DATA.NUM_FRAMES
    return model
