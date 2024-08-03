from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms


def print_data_info(inputs):
    tensor = inputs['pixel_values']
    print('Device:{}, shape:{}, max:{:.4f}, min: {:.4f}'.format(tensor.device,
            tensor.shape, tensor.max(), tensor.min()))


def plip(device, gpu_num):
    processor = CLIPProcessor.from_pretrained('vinid/plip')
    model = CLIPModel.from_pretrained('vinid/plip').to(device)

    # if gpu_num > 1:
    #     model = torch.nn.parallel.DataParallel(model)
    model.eval()

    def func(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # print_data_info(inputs)
        img_embed = model.get_image_features(**inputs)
        # img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        return img_embed
    return func


def plip_transformers():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    return trnsfrms_val

if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import numpy as np
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = torch.from_numpy(np.array(image))
    image = torch.stack([image, image], dim=0)
    device = torch.device('cuda:1')
    model = plip(device, 1)
    outputs = model(image)
    print(outputs.min(), outputs.max())
    print(outputs.shape)
    print()