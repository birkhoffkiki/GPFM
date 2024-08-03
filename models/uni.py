# https://huggingface.co/MahmoodLab/UNI
import timm
from torchvision import transforms
import torch
    
    
def get_uni_trans():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform



def get_uni_model(device):
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load('models/ckpts/uni.bin', map_location="cpu"), strict=True)
    model.eval()
    
    return model.to(device)