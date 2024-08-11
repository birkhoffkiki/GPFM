# %%
import sys
sys.path.append('/storage/Pathology/codes/EasyMIL')
import seaborn
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from models.dinov2 import build_model
from PIL import Image
import numpy as np
import cv2
import seaborn as sns

# %%
DisFM, _ = build_model('cuda:5', 1, 'distill_87499', '/storage/Pathology/codes/EasyMIL/models/ckpts/distill_87499.pth')

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

# %%
root='/home/jmabq/data/BACH/Invasive/iv093.tif'
raw_img = Image.open(root)

# %%
tsize = (1344, 1344)
raw_img = transforms.CenterCrop((tsize))(raw_img)

all_size = [(224, 224), (448, 448), (896, 896), (1344, 1344)]

for index, size in enumerate(all_size):

    vis_img = raw_img.resize(size)
    vis_img.save('vis.png')
    img = normalize(vis_img)[None].cuda(5)

    with torch.no_grad():
        result = DisFM(img, is_training=True)
    cls_token = result['x_norm_clstoken'][None]
    patch_tokens = result['x_norm_patchtokens']
    
    similarity = torch.nn.functional.mse_loss(cls_token, patch_tokens, reduction='none').mean(dim=-1)
    # to 0-1
    similarity = (similarity - similarity.min())/(similarity.max() - similarity.min())
    similarity = similarity.view([i//14 for i in size]).cpu().numpy()
    
    plt.figure(index)
    plt.imshow(vis_img)
    similarity = np.array(Image.fromarray(similarity).resize(size, resample=0))
    # plt.imshow(similarity, alpha=0.3, cmap='hot')
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    seaborn.heatmap(similarity, alpha=0.7, cmap=cmap, vmin=0, vmax=1)
    # plt.imshow(similarity, alpha=0.5, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig('{} vis.png'.format(index), dpi=250)
    