# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def get_img_path(self, index):
        return self._dataset.data_items[index]
    
    def get_image_data(self, index: int) -> bytes:
        return self._dataset[index][0]

    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset[index][1][1]
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)


class SimpleDataset(Dataset):
    def __init__(self, feat, label) -> None:
        super().__init__()
        self.feat = feat
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index) -> Any:
        return self.feat[index], (index, self.label[index])
        