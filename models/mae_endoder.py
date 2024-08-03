from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
from models import models_mae


def print_data_info(inputs):
    # tensor = inputs['pixel_values']
    tensor = inputs
    print('Device:{}, shape:{}, max:{:.4f}, min: {:.4f}'.format(tensor.device,
            tensor.shape, tensor.max(), tensor.min()))
    

def mae_transform(input_size):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    return transforms.Compose([
        # transforms.RandomResizedCrop(input_size, scale=(0.25, 1), interpolation=3, ratio=(1.0, 1.0)),  # 3 is bicubic
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(input_size, interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def mae_pretrained_model(device, gpu_num, model, ckpt, input_size=224):
    model=models_mae.__dict__[model](norm_pix_loss=False)
    # checkpoint = model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'], strict=False)
    checkpoint = torch.load(ckpt, map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    if gpu_num > 1:
        model = torch.nn.parallel.DataParallel(model)
    model.eval()
    return model


if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import numpy as np
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    # image = torch.from_numpy(np.array(image))
    image = mae_transform(224)(image).cuda(0)
    image = torch.stack([image, image], dim=0)
    model = mae_pretrained_model('cuda:0', 1, 'mae_vit_large_patch16', '/jhcnas3/Pathology/outputs_vit_l_resume/checkpoint-1-40000.pth', input_size=224)
    outputs = model(image)
    print(outputs.shape)
