import torch
import timm
from transformers import ViTModel
from conch.open_clip_custom import create_model_from_pretrained
from torch.nn import functional as F


class Phikon(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        self.dim = 768
        for name, param in self.model.named_parameters():
            param.requires_grad = False
    
    def forward(self, img):
        # after norm
        output = self.model(img)
        cls_token = output.last_hidden_state[:, :1, :]
        patch_token = output.last_hidden_state[:, 1:, :]
        return {'cls_token': cls_token, 'patch_token': patch_token}
    

class UNI(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = 1024
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        msg = model.load_state_dict(torch.load('dinov2/weights/uni.bin', map_location="cpu"), strict=True)
        print("Loading weights for UNI", msg)
        self.model = model
    
    def forward(self, img):
        # after norm
        features =  self.model.forward_features(img)
        patch_token = features[:, 1:, :]
        cls_token = self.model.forward_head(features)[:, None, :]
        return {'cls_token': cls_token, 'patch_token': patch_token}


class CONCH(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 512
        coca, _ = create_model_from_pretrained('conch_ViT-B-16', "dinov2/weights/conch.pth")
        self.model = coca.visual
            
    def forward(self, img):
        # https://huggingface.co/MahmoodLab/CONCH
        # image_embs = self.model.encode_image(img, proj_contrast=False, normalize=False)
        cls_token, patch_token = self.model.forward(img)
        cls_token = cls_token[:, None, :]
        # norm
        cls_token = F.normalize(cls_token, dim=-1)
        patch_token = F.normalize(patch_token, dim=-1)
        # print(cls_token.shape)
        # print(patch_token.shape)
        return {'cls_token': cls_token, 'patch_token': patch_token}



if __name__ == '__main__':
    img = torch.randn(12, 3, 224, 224)
    model = Phikon()
    output = model(img)
    print(output.shape)
