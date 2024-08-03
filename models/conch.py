# pip install git+https://github.com/Mahmoodlab/CONCH.git

from conch.open_clip_custom import create_model_from_pretrained
import torch


def get_conch_model(device):
    model, _ = create_model_from_pretrained('conch_ViT-B-16', "models/ckpts/conch.pth", device=device)

    def func(image):
        # get the features
        with torch.no_grad():
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
            return image_embs
        
    return func


def get_conch_trans():
    _, preprocess = create_model_from_pretrained('conch_ViT-B-16', "models/ckpts/conch.pth")
    return preprocess