# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from .extended import ExtendedVisionDataset
import os
from PIL import Image
import json
import cv2

logger = logging.getLogger("dinov2")


class PathologyDataset(ExtendedVisionDataset):
    def __init__(self,
        root: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, None, transform, target_transform)
        self.image_paths = self.get_all_files(root)
        try:
            self.image_patch = cv2.imread(self.image_paths[0])[..., ::-1]
        except:
            raise RuntimeError()
        
        self.transformers = transform
        self.file_handle = open('./failed_path.txt', 'a')
    
    def get_all_files(self, root):
        paths = []
        with open(root, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                for name in v:
                    p = os.path.join(k, name)
                    paths.append(p)
        return paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def try_fast_disk(self, p: str):
        new_p = p.replace('/project/vcompath/storage/Pathology', '/scratch/vcompath')
        if os.path.exists(new_p):
            return new_p
        else:
            return p

    def __getitem__(self, index):
        p = self.image_paths[index]
        p = self.try_fast_disk(p)
        try:
            # img = cv2.imread(p)[..., ::-1] #BGR to RGB
            # img = Image.fromarray(img)
            img = Image.open(p)
            if self.transformers is not None:
                img = self.transformers(img)
        except:
            img = Image.fromarray(self.image_patch.copy())
            if self.transformers is not None:
                img = self.transformers(img)
            self.file_handle.write(p + '\n')
            self.file_handle.flush()
        return img, None
