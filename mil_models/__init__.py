import torch


def find_mil_model(model_name, in_dim, n_classes, drop_out, task_type, dataset_name=None):
    """Define how to init MIL model Here

    Args:
        model_name (string): the mil model name
        in_dim (int): the dimention of feature, e.g. 1024
        n_classes (int): class number of dataset
        drop_out (float): drop_out ratio
        task (string): subtyping or survial

    Raises:
        NotImplementedError: _description_

    Returns:
        MIL Model: _description_
    """
    if '||' in in_dim:
        in_dim = [int(i) for i in in_dim.split('||')]
    else:
        in_dim = int(in_dim)
        
    
    if model_name.lower() == 'dtfd':
        from .dtfd.dtfd_mil import DTFD_Model
        model = DTFD_Model(in_dim, num_cls=n_classes, droprate=drop_out, droprate_2=drop_out, task=task_type)
    
    elif model_name.lower() == 'wikg':
        from .wikg import WiKG
        model = WiKG(in_dim, n_classes=n_classes, task_type=task_type)
        
    elif model_name.lower() == 'att_mil':
        from .att_mil import DAttention
        model = DAttention(in_dim, n_classes, dropout=True, act='relu', task_type=task_type)
    
    elif model_name.lower() == 'mean_mil':
        from .mean_max_mil import MeanMIL
        model = MeanMIL(in_dim, n_classes, dropout=True, act='relu', task_type=task_type)
        
    elif model_name.lower() == 'max_mil':
        from .mean_max_mil import MaxMIL
        model = MaxMIL(in_dim, n_classes, dropout=True, act='relu', task_type=task_type)
        
    elif model_name.lower() == 'trans_mil':
        from .trans_mil import TransMIL
        model = TransMIL(in_dim, n_classes, dropout=drop_out, act='relu', task_type=task_type)
        
    elif model_name.lower() == 'ds_mil':
        from .ds_mil import FCLayer, BClassifier, MILNet
        dropout = 0.25 if drop_out else 0
        i_classifier = FCLayer(in_size=in_dim, out_size=n_classes)
        b_classifier = BClassifier(input_size=in_dim, output_class=n_classes, dropout_v=dropout)
        model = MILNet(i_classifier, b_classifier, task_type=task_type)
    
    elif model_name.lower() == 'clam_sb':
        from .model_clam import CLAM_SB
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda()
        model = CLAM_SB(in_dim, n_classes=n_classes, dropout=drop_out, task_type=task_type, instance_loss_fn=instance_loss_fn)    
    
    elif model_name.lower() == 'simple':
        from .simple_mil import Simple
        model = Simple(in_dim=in_dim, n_classes=n_classes, act='gelu', task=task_type)

    elif model_name.lower() == 'moe':
        from .moe_mil import MoE
        if dataset_name == 'BRACS':
            samples_per_cls = [66, 60, 54, 61, 56, 83, 57]
        elif dataset_name == 'TCGA_BRCA_subtyping':
            samples_per_cls = [401, 386]
        else:
            samples_per_cls = []
            
        print('Sample_per_cls has been set:', samples_per_cls)
        model = MoE(in_dim=in_dim, n_classes=n_classes, task=task_type, samples_per_cls=samples_per_cls)
    
    elif model_name.lower() == 'moe_a2o':
        from .moe_all2one import MoE
        if dataset_name == 'BRACS':
            samples_per_cls = [66, 60, 54, 61, 56, 83, 57]
        elif dataset_name == 'TCGA_BRCA_subtyping':
            samples_per_cls = [401, 386]
        else:
            samples_per_cls = []
            
        print('Sample_per_cls has been set:', samples_per_cls)
        model = MoE(in_dim=in_dim, n_classes=n_classes, task=task_type, samples_per_cls=samples_per_cls) 

    elif model_name.lower() == 'mamba':
        from .mamba_moe.model import MoE
        model = MoE(in_dim=in_dim, n_classes=n_classes, task=task_type) 

    #*
    elif model_name.lower() == 'hat_encoder_256_32':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, dropout = 0.25, topk = 32, k = 256, task_type=task_type)
        _load_checkpoint(model, model.model_path_256_32)
    elif model_name.lower() == 'hat_encoder_256_32_nomem':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, dropout = 0.25, topk = 32, k = 256, task_type=task_type)
        _load_checkpoint(model, model.model_path_256_32)
    elif model_name.lower() == 'hat_encoder_256_256':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, dropout = 0.25, topk = 256, k = 256, task_type=task_type)
        _load_checkpoint(model, model.model_path_256_256)
    elif model_name.lower() == 'hat_encoder_512_512':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_512_512)
    elif model_name.lower() == '96_hat_encoder_512_512':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, region_size = 96, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_96_512_512)
    elif model_name.lower() == '96_hat_encoder_512_512_nomem':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, region_size = 96, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_96_512_512)
    elif model_name.lower() == '512_hat_encoder_512_512':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, region_size = 512, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_512_512_512)
    elif model_name.lower() == '128_hat_encoder_512_512':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, region_size = 128, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_128_512_512)
    elif model_name.lower() == '256_hat_encoder_512_512':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, region_size = 256, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_256_512_512)
    elif model_name.lower() == '64_hat_encoder_512_512':
        def _load_checkpoint(model, load_path):
            load_path = str(load_path)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        from .hat_model import HAT_model
        model = HAT_model(in_dim = in_dim, n_classes = n_classes, region_size = 64, dropout = 0.25, topk = 512, k = 512, task_type=task_type)
        _load_checkpoint(model, model.model_path_64_512_512)
    #*
    else:
        raise NotImplementedError(f'{model_name}is not implemented ...')
    return model


if __name__ == '__main__':
    model = find_mil_model('wsi_agg_sup_depth1', 1024, 3, 0.1, 'subtyping')
    x = model.state_dict()
    print(x.keys())
