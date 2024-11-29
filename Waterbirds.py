#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Callable, Tuple, Generator, Union
from collections import OrderedDict
from torch.utils.data import ConcatDataset
import pandas as pd


    


data_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Waterbirds(Dataset):
    def __init__(self, env: str, data_dir: str = "./data/waterbirds", transform = data_transform, metadata_filename: str = "metadata.csv", return_index: bool = False):
        self.data_dir:          str  = data_dir
        self.env:               str  = env
        self.metadata_filename: str  = metadata_filename
        self.return_index:      bool = return_index
        self.num_classes = 2

        self.env_to_split = {
            "train": 0,
            "val":   1,
            "test":  2
        }

        self.transform = transform
        self.metadata_path = os.path.join(self.data_dir, self.metadata_filename)

        metadata_csv = pd.read_csv(self.metadata_path)
        metadata_csv = metadata_csv.query(f"split == {self.env_to_split[self.env]}")

        self.samples = {}
        self.files_count = 0
        for i, (_, sample_info) in enumerate(metadata_csv.iterrows()):
            self.samples[i] = {
                "image_path":  os.path.join(self.data_dir, sample_info["img_filename"]),
                "class_label": int(sample_info["y"]),
                "bias_label": -1 if int(sample_info["y"]) != int(sample_info["place"]) else 1,
                "all_attrs": list((str(e) for e in sample_info))
            }
            self.files_count += 1

    def __len__(self) -> int:
        return self.files_count

    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        image = self.transform(Image.open(self.samples[index]["image_path"]))
        class_label = self.samples[index]["class_label"]
        bias_label  = self.samples[index]["bias_label"]

        if self.return_index:
            return image, class_label, bias_label, index

        return image, class_label, bias_label
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i][2]

    def __repr__(self) -> str:
        return f"Waterbirds(env={self.env}, bias_amount=Fixed, num_classes={self.num_classes})"