# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
import copy

from downstream_tasks.loader import make_data_loader, make_dataset
from downstream_tasks.metrics import build_linear_metric
from downstream_tasks.utils import evaluate, ModelInference
from downstream_tasks.datasets import SimpleDataset


def get_args_parser(
    description: Optional[str] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size (per GPU)")

    return parser



class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, cls_tokens):
        return self.linear(cls_tokens)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, cls_mapping):
        super().__init__()

    def forward(self, samples, targets):
        return {'preds': samples, 'target': targets}



@torch.no_grad()
def evaluate_linear_classifiers(
    linear_classifiers,
    data_loader,
    training_num_classes,
    metric,
    class_mapping=None,
):
    print("running validation !")
    metrics = {'linear_prob': metric}
    postprocessors = {'linear_prob': LinearPostprocessor(linear_classifiers, class_mapping)}

    results_dict, raw_result = evaluate(linear_classifiers, data_loader, postprocessors, metrics, torch.cuda.current_device())
    print(results_dict)
    
    try:
        auc_value = results_dict['linear_prob']['AUC'].item()
    except:
        auc_value = None
        
    final_dict = {}
    for knn_name, result in results_dict.items():
        temp_dict = {}
        for k, v in result.items():
            temp_dict[k] = v.item()
        final_dict[knn_name] = temp_dict
    return final_dict, auc_value, raw_result


def eval_linear(
    *,
    epoches,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    training_num_classes,
    val_class_mapping=None,
):
    iteration = 0
    best_auc_value = 0
    best_weight = None
    best_ep = 0
    early_stop_counter = 0
    for ep in range(epoches):
        for features, (index, labels) in train_data_loader:
            iteration += 1
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = linear_classifiers(features)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()
            scheduler.step()

            # log
            if iteration % 10 == 0:
                print("loss: {:.4f}, lr: {:.6}".format(loss.item(), optimizer.param_groups[0]["lr"]))

        metrics = build_linear_metric(training_num_classes, bootstrap=False)
        results_dict, auc_value, raw_result = evaluate_linear_classifiers(
            linear_classifiers=linear_classifiers,
            data_loader=val_data_loader,
            training_num_classes=training_num_classes,
            class_mapping=val_class_mapping,
            metric=metrics
        )
        early_stop_counter += 1
        if auc_value > best_auc_value:
            best_auc_value = auc_value
            best_ep = ep
            # reset early stop counter 
            early_stop_counter = 0
            best_weight ={k: v.clone() for k, v in linear_classifiers.state_dict().items()}
            print('New model detected, saving result, ep={}, auc={}'.format(ep, auc_value))
            with open(metrics_file_path, "a") as f:
                f.write(f"ep: {ep}\n")
                for k, v in results_dict.items():
                    f.write(json.dumps({k: v}) + "\n")
                f.write("\n")
        if early_stop_counter >= 100:
            return best_weight, best_ep
        
    return best_weight, best_ep


def run_eval_linear(
    data_dir,
    output_dir,
    batch_size,
    epochs,
):
    torch.manual_seed(0)
    train_data = torch.load(os.path.join(data_dir, 'train.pt'))
    val_data = torch.load(os.path.join(data_dir, 'val.pt'))
    if os.path.exists(os.path.join(data_dir, 'test.pt')):
        test_data = torch.load(os.path.join(data_dir, 'test.pt'))
    else:
        test_data = val_data
    
    training_num_classes = train_data['label'].max().item() + 1
    feat_dim = train_data['feature'].shape[-1]
    
    linear_classifiers = LinearClassifier(feat_dim, training_num_classes).cuda()
    optimizer = torch.optim.AdamW(linear_classifiers.parameters(), lr=5e-4, weight_decay=1e-5)

    train_set = SimpleDataset(train_data['feature'], train_data['label'])
    val_set = SimpleDataset(val_data['feature'], val_data['label'])
    train_loader = make_data_loader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = make_data_loader(val_set, batch_size=batch_size, num_workers=0, shuffle=False)
    max_iter = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=1e-5)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    # delete if exists
    if os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)
    
    best_weight, ep = eval_linear(
        epoches=epochs,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        training_num_classes=training_num_classes,
    )
        
    metric = build_linear_metric(training_num_classes, bootstrap=True)
    msg = linear_classifiers.load_state_dict(best_weight)
    torch.save(best_weight, os.path.join(output_dir, 'ep={}model.ckpt'.format(ep)))
    
    print('best epoch:{}'.format(ep), msg)
    test_set = SimpleDataset(test_data['feature'], test_data['label'])
    test_loader = make_data_loader(test_set, batch_size=1024, num_workers=0, shuffle=False)
    result_dict, _, raw_result = evaluate_linear_classifiers(linear_classifiers=linear_classifiers, data_loader=test_loader, 
                    training_num_classes=training_num_classes, metric=metric)
    torch.save(raw_result, os.path.join(output_dir, 'ep={}_predictions.ckpt'.format(ep)))
    print('Saving test result')
    with open(metrics_file_path, "a") as f:
        f.write("Test result")
        for k, v in result_dict.items():
            f.write(json.dumps({k: v}) + "\n")
        f.write("\n")




if __name__ == "__main__":
    description = "linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(os.path.join(args.output_dir))
        
    run_eval_linear(data_dir=args.data_dir, output_dir=args.output_dir,
        batch_size=args.batch_size, epochs=args.epochs)
