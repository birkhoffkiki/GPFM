from models.uni import get_uni_model
from torchvision import transforms
import torch

def get_mmstar_trans():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform

def get_mmstar_model(device):
    model = get_uni_model(device)
    model_ckpt = '/jhcnas4/wangyihui/MultiModal/stage3_checkpoints/2024-05-07_19-38-56/model_ema_epoch_249.pth'
    print("load model from {}".format(model_ckpt))
    try:
        ckpt = torch.load(model_ckpt, map_location=torch.device('cpu'))
        ckpt = ckpt['model_state_dict']
        updated_ckpt = {}
        for k, v in ckpt.items():
            if 'vision_encoder' in k:
                k = k.replace('vision_encoder.', '')
            if 'module.' in k:
                k = k.replace('module.', '')
            updated_ckpt[k] = v
        load_status = model.load_state_dict(updated_ckpt, strict=False)
            
        # Check for missing and unexpected keys
        missing_keys = load_status.missing_keys
        unexpected_keys = load_status.unexpected_keys
        if missing_keys:
            print("Warning: Missing keys in state_dict:", missing_keys)
        if unexpected_keys:
            print("Warning: Unexpected keys in state_dict:", unexpected_keys)
        if not missing_keys and not unexpected_keys:
            print("Checkpoint loaded successfully from {}.".format(model_ckpt))
    except FileNotFoundError:
        print("Checkpoint file not found at {}".format(model_ckpt))
    except KeyError as e:
        print("Invalid checkpoint format, key {} not found.".format(e))
    except Exception as e:
        print("An error occurred while loading the checkpoint: {}".format(e))
    
    model.eval()
    return model.to(device)