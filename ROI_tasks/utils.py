# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Dict, Optional
import torch
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm

from downstream_tasks.loader import make_data_loader
from downstream_tasks.datasets import DatasetWithEnumeratedTargets


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)

class ModelInference(torch.nn.Module):
    def __init__(self, model, autocast_ctx) -> None:
        super().__init__()
        self.model = model
        self.autocast_ctx = autocast_ctx
    
    def forward(self, imgs):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.mode(imgs)
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader, 
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    raw_result = {'logits': [], 'labels': []}
    
    for samples, (_, targets) in tqdm(data_loader, total=len(data_loader)):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        raw_result['logits'].append(outputs.cpu())
        raw_result['labels'].append(targets.cpu())
        
        if criterion is not None:
            loss = criterion(outputs, targets)
            print('loss:', loss)
            
        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            # print(metric_inputs.keys())
            # metric.update(metric_inputs['preds'].cpu(), metric_inputs['target'].cpu())
            metric.update(metric_inputs['preds'], metric_inputs['target'])

    stats = {k: metric.compute() for k, metric in metrics.items()}
    return stats, raw_result



def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    features, all_labels = [], []
    for index, (samples, (_, labels_rank)) in enumerate(data_loader):
        print('Extracting feature progresss:{}/{}'.format(index+1, len(data_loader)), flush=True)
        samples = samples.cuda()
        features_rank = model(samples)

        features.append(features_rank.to(gather_device))
        all_labels.append(labels_rank)

    features = torch.cat(features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Features shape: {tuple(features.shape)}")
    print(f"Labels shape: {tuple(all_labels.shape)}")
    assert torch.all(all_labels > -1)
    return features, all_labels, []
