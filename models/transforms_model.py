from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch

def print_data_info(inputs):
    tensor = inputs['pixel_values']
    print('Device:{}, shape:{}, max:{:.4f}, min: {:.4f}'.format(tensor.device,
            tensor.shape, tensor.max(), tensor.min()))


def vit_large_patch16_224_21k(device, gpu_num):
    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k').to(device)
    if gpu_num > 1:
        model = torch.nn.parallel.DataParallel(model)
    model.eval()

    def func(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        print_data_info(inputs)
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 0, :]   # Get CLS embedding
    return func


def vit_base_patch16_224_21k(device, gpu_num):
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    if gpu_num > 1:
         model = torch.nn.parallel.DataParallel(model)
    model.eval()

    def func(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        print_data_info(inputs)
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 0, :]   # Get CLS embedding
    return func

def vit_huge_patch14_224_21k(device, gpu_num):
    processor = ViTImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
    model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k').to(device)
    if gpu_num > 1:
        model = torch.nn.parallel.DataParallel(model)
    model.eval()

    def func(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        print_data_info(inputs)
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 0, :]   # Get CLS embedding
    return func


if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import numpy as np
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = torch.from_numpy(np.array(image))
    image = torch.stack([image, image], dim=0)
    model = vit_huge_patch14_224_21k('cpu', 0)
    outputs = model(image)
    print(outputs.shape)
    print()
