import sys
sys.path.append('/home/jmabq/dinov2')

import torch
import safetensors
from dinov2.models.vision_transformer import vit_huge


openai_clip_vith = '/home/jmabq/dinov2/dinov2/weights/timm_vit_huge_patch14_clip_224.pth'
new_path = '/home/jmabq/dinov2/dinov2/weights/timm_vit_huge_patch14_dinov2.pth'

ckpt = torch.load(openai_clip_vith)
new_ckpt = {}

for k, v in ckpt.items():
    if 'blocks' in k:
        print('old k:', k)
        items = k.split('.')
        ch = str(int(int(items[1]) // 8))
        items.insert(1, ch)
        k = '.'.join(items)
        print('new k:', k)
    new_ckpt[k] = v

torch.save(new_ckpt, new_path)
        
model = vit_huge(block_chunks=4, init_values=0.0001)
msg = model.load_state_dict(new_ckpt, strict=False)

print(msg)