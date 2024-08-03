import torch
import math
from torch.nn import functional

ckpt_path = '/home/jmabq/dinov2/dinov2/weights/dinov2_vitg14_pretrain.pth'
new_path = '/home/jmabq/dinov2/dinov2/weights/dinov2_vitg14_pretrain_unblock.pth'

def resize_pos_embedding(pos_embed):
    # (1, N, d)
    size=(224//14, 224//14)
    print(size)
    class_pos_embed = pos_embed[:, 0:1]
    patch_pos_embed = pos_embed[:, 1:]
    b, n, d = patch_pos_embed.shape
    s = int(math.sqrt(n))
    patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
    patch_pos_embed = patch_pos_embed.view(1, d, s, s)
    patch_pos_embed = functional.interpolate(patch_pos_embed, size=size, mode='bilinear')
    print('shape:', patch_pos_embed.shape)
    patch_pos_embed = patch_pos_embed.view(1, d, -1)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
    new_pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)
    return new_pos_embed


new_ckpt = {}
factor=10
ckpt = torch.load(ckpt_path)
for k, v in ckpt.items():
    if 'pos_embed' in k:
        v = resize_pos_embedding(v)
        
    if 'blocks' in k:
        print('old k:', k)
        items = k.split('.')
        ch = str(int(int(items[1]) // factor))
        items.insert(1, ch)
        k = '.'.join(items)
        print('new k:', k)
    new_ckpt[k] = v

torch.save(new_ckpt, new_path)
        