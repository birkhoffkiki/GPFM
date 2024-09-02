import torch
import torch.nn as nn
import torch.nn.functional as F
from .mil import BaseMILModel
from mil_models.hat_modules.hat_module_all import BaseCMN, HATPooler

class HAT_model(BaseMILModel):
    def __init__(self, in_dim = 1024, n_classes = 2, dropout = 0.1, region_size = 384, topk = 256, k = 256, task_type = 'subtyping'):
        super(HAT_model, self).__init__(task=task_type)
        self.encoder_decoder = BaseCMN(dropout = dropout, topk = topk, region_size = region_size, k = k)
        self.encoder_layout = {
                'num_heads': 8,
                'd_model': 512,
                'd_ff': 512,
                'region_size': region_size,
                'dropout': dropout,
                'pooling': 'attentive',
                'num_layers': 2,
                '0': {
                    'region_encoder': True,
                    'WSI_encoder': True,
                    'first_layer': True
                },
                '1': {
                    'region_encoder': True,
                    'WSI_encoder': False,
                    'first_layer': False
                },
            }
        self.wsi_mapping = nn.Linear(in_dim, 2048)
        self.pooler = HATPooler(self.encoder_layout)
        self.classifier = nn.Linear(512, n_classes)
        
        self.model_path_256_32 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_384_layer_2_lgl_attentive_uniform_sampling_mem/max-len-100/model_best.pth'
        self.model_path_512_512 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_384_layer_2_lgl_attentive_uniform_sampling_mem_512/max-len-100/model_best.pth'
        self.model_path_256_256 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_384_layer_2_lgl_attentive_uniform_sampling_mem_256/max-len-100/model_best.pth'
        self.model_path_96_512_512 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_96_layer_2_lgl_attentive_uniform_sampling_mem_512/max-len-100/model_best.pth'
        self.model_path_512_512_512 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_512_layer_2_lgl_attentive_uniform_sampling_mem_512/max-len-100/model_best.pth'
        self.model_path_128_512_512 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_128_layer_2_lgl_attentive_uniform_sampling_mem_512/max-len-100/model_best.pth'
        self.model_path_64_512_512 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_64_layer_2_lgl_attentive_uniform_sampling_mem_512/max-len-100/model_best.pth'
        self.model_path_256_512_512 = '/storage/Pathology/wsi-report/results/test_memory/dinov2_vitl/hi_transformer_region_256_layer_2_lgl_attentive_uniform_sampling_mem_512/max-len-100/model_best.pth'
        
    def forward(self, x, **kwargs):
        x = self.wsi_mapping(x)
        x = x.unsqueeze(0)
        features = self.encoder_decoder(x, mode = 'forward')
        # Pooling region-representations to obatin the global representation
        pooled_features = self.pooler(features)
        # classification
        logits = self.classifier(pooled_features)

        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)

        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        
        return outputs
    
def _load_checkpoint(model, load_path):
    load_path = str(load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])
        
if __name__ == '__main__':
    device = torch.device("cuda")
    data = torch.randn(1, 1000, 1024)
    data = data.to(device)
    hat = HAT_model(topk=32)
    hat = hat.to(device)
    _load_checkpoint(hat, hat.model_path_256_32)
    out = hat(data)