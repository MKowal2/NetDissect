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
        cfg = load_config(model_args, 'loader/models/slowfast/configs/Kinetics/c2/SLOW_8x8_R50.yaml')
        cfg = assert_and_infer_cfg(cfg)

        # if settings.DATASET == 'Diving48':
        #     cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
        #     cfg.TEST.CHECKPOINT_FILE_PATH = 'models/ar_models/checkpoints/slowonly_8x8_2gpu_run1/checkpoint_epoch_00100.pyth'
        #     cfg.MODEL.NUM_CLASSES = 48
        #     if len(args.checkpoint) > 0:
        #         cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
        #         cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
        #         cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'

        model = build_model(cfg)
        cu.load_test_checkpoint(cfg, model)
        settings.SAMPLING_RATE = cfg.DATA.SAMPLING_RATE
        settings.NUM_FRAMES = cfg.DATA.NUM_FRAMES
    else:
        print('Model selected not available')
        raise NotImplementedError
    return model
