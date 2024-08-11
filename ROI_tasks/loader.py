# Copyright (c) Meta Platforms, Inc. and affiliates.
#
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def make_dataset(
    *,
    dataset_str: str,
    transform = None,
    target_transform = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    print(f'using dataset: "{dataset_str}"')

    dataset_name, phase = dataset_str.split(":")
    if dataset_name.lower() == 'crc-100k':
        from downstream_tasks.datasets.crc100k import DataSet
    elif dataset_name.lower() == 'bach': 
        from downstream_tasks.datasets.bach import DataSet
    elif dataset_name.lower() == 'ccrcc-tcga_hel': 
        from downstream_tasks.datasets.ccrcc_tcga_hel import DataSet
    elif dataset_name.lower() == 'unitopatho': 
        from downstream_tasks.datasets.unitopatho import DataSet
    elif dataset_name.lower() == 'pancancer-tcga': 
        from downstream_tasks.datasets.pancancer_tcga import DataSet
    elif dataset_name.lower() == 'pancancer-til': 
        from downstream_tasks.datasets.pancancer_til import DataSet
    elif dataset_name.lower() == 'crc-msi': 
        from downstream_tasks.datasets.crc_msi import DataSet
    elif dataset_name.lower() == 'lc2500': 
        from downstream_tasks.datasets.lc2500 import DataSet
    elif dataset_name.lower() == 'esca': 
        from downstream_tasks.datasets.esca import DataSet
    elif dataset_name.lower() == 'pcam': 
        from downstream_tasks.datasets.pcam import DataSet
    elif dataset_name.lower() == 'wsss4luad': 
        from downstream_tasks.datasets.wsss4luad import DataSet
    elif dataset_name.lower() == 'breakhis': 
        from downstream_tasks.datasets.breakhis import DataSet
    elif dataset_name.lower() == 'chaoyang': 
        from downstream_tasks.datasets.chaoyang import DataSet
    elif dataset_name.lower() == 'lysto': 
        from downstream_tasks.datasets.lysto import DataSet
    elif dataset_name.lower() == 'dryad': 
        from downstream_tasks.datasets.dryad import DataSet
    else:
        raise NotImplementedError(f'{dataset_name}')
    
        
    dataset = DataSet(transformer=transform, phase=phase)

    print(f"# of dataset samples: {len(dataset):,d}")

    return dataset


def make_data_loader(dataset=None, batch_size=256, num_workers=8, shuffle=True, use_ddp=False, collate_fn=None):
    if use_ddp:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader
