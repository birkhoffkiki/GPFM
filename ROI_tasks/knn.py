# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
from typing import List, Optional

import torch
from torch.nn.functional import one_hot, softmax

from downstream_tasks.loader import make_data_loader
from downstream_tasks.datasets import SimpleDataset
from downstream_tasks.metrics import build_knn_metric
from downstream_tasks.utils import evaluate


logger = logging.getLogger("downstream_tasks")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[1, 20],
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
    )
    return parser


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.device = device
        self.train_features = train_features.T.to(device)   # (c, N)
        self.train_features = torch.nn.functional.normalize(self.train_features, dim=0)
        self.candidates = train_labels.to(device) # (N)
        
        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _similarity_for_rank(self, feature):
        # feat: (n, c)
        feature = torch.nn.functional.normalize(feature, dim=1)

        similarity = torch.mm(feature, self.train_features) # (n, N)
        candidate_labels = self.candidates.expand(len(similarity), -1) # (n, N)
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True, dim=-1)
        neighbors_labels = torch.gather(candidate_labels, 1, indices)
        return topk_sims, neighbors_labels

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)
        topk_sims, neighbors_labels = self._similarity_for_rank(features_rank)
        batch_size = neighbors_labels.shape[0]
        
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes), # (b, nn, class_num)
            topk_sims_transform.view(batch_size, -1, 1),    # (b, nn, 1)
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def eval_knn(train_dataset, val_dataset, nb_knn, temperature, n_per_class_list=[-1], n_tries=1,
             batch_size=256):

    train_features, train_labels = train_dataset['feature'], train_dataset['label']
    print(f"Train features created, shape {train_features.shape}.")
    val_features, val_labels = val_dataset['feature'], val_dataset['label']
    print(f"Val features created, shape {val_features.shape}.")
    val_dataset = SimpleDataset(val_features, val_labels)
    val_loader = make_data_loader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    num_classes = train_labels.max() + 1
    print('Class number:', num_classes)
    metric_collection = build_knn_metric(num_classes=num_classes)
    device = torch.cuda.current_device()
    
    partial_module = partial(KnnModule, T=temperature, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(knn_module_dict)

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    results_dict = evaluate(model_with_knn, val_loader, postprocessors, metrics, device)

    print(results_dict)
    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict


def eval_knn_with_model(
    output_dir,
    data_dir,
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    batch_size=256,
    n_per_class_list=[-1],
    n_tries=1,
):
    train_dataset = torch.load(os.path.join(data_dir, 'train.pt'))
    val_dataset = torch.load(os.path.join(data_dir, 'val.pt'))
    
    results_dict_knn = eval_knn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        nb_knn=nb_knn,
        temperature=temperature,
        batch_size=batch_size,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
    )

    results_dict = {}
    for knn_name, result in results_dict_knn.items():
        temp_dict = {}
        for k, v in result.items():
            temp_dict[k] = v.item()
        knn_name = '-'.join([str(i) for i in knn_name])
        results_dict[knn_name] = temp_dict
        

    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "w") as f:
        for k, v in results_dict.items():
            f.write(json.dumps({k: v}) + "\n")
    return results_dict


if __name__ == "__main__":
    description = "k-NN evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_knn_with_model(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        batch_size=args.batch_size,
        n_per_class_list=args.n_per_class_list,
        n_tries=args.n_tries,
    )