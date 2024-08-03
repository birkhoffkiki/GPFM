import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from models import get_custom_transformer, get_model

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

def has_Phikon():
    model = get_model('phikon')

def has_ctranspath():
    HAS_CTRANSPATH = False
    CTRANSPATH_CKPT_PATH = ''
    # check if CTRANSPATH_CKPT_PATH is set, catch exception if not
    try:
        # check if CTRANSPATH_CKPT_PATH is set
        if 'CTRANSPATH_CKPT_PATH' not in os.environ:
            raise ValueError('CTRANSPATH_CKPT_PATH not set')
        HAS_CTRANSPATH = True
        CTRANSPATH_CKPT_PATH = os.environ['CTRANSPATH_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_CTRANSPATH, CTRANSPATH_CKPT_PATH

def has_plip():
    model = get_model('plip')

def has_DisFM():
    model = get_model('distill_87499')

        
def get_encoder(model_name, device, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'phikon':
        model = get_model('phikon', device, torch.cuda.device_count())
    elif model_name == 'ctranspath':
        HAS_CTRANSPATH, CTRANSPATH_CKPT_PATH = has_ctranspath()
        assert HAS_CTRANSPATH, 'CTRANSPATH is not available'
        model = get_model('ctranspath', device, torch.cuda.device_count())
    elif model_name == 'plip':
        model = get_model('plip', device, torch.cuda.device_count())
    elif model_name == 'disfm':
        model = get_model('distill_87499', device, torch.cuda.device_count())
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    if model_name in ['plip', 'phikon', 'disfm', 'ctranspath']:
        if model_name == 'disfm':
            model_name = 'distill_87499'
        img_transforms = get_custom_transformer(model_name)
    else:
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms