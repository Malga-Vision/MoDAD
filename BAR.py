#!/usr/bin/env python3

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Generator, Union
import os
import shutil
from os import path
from PIL import Image
from sklearn.model_selection import train_test_split

class BAR(Dataset):
    eval_transform = transforms.Compose([ # TODO: Check if people applies particular transforms 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    classes_str_to_idx = {
        "climbing": 0,
        "diving":   1,
        "fishing":  2,
        "racing":   3,
        "throwing": 4,
        "pole vaulting": 5,
    }


    def __init__(self, data_dir="./data/BAR-master", env="train", return_index=False) -> None:
        self.data_dir = data_dir
        self.transform = BAR.eval_transform if env == "test" else BAR.train_transform
        
        if env == "val":
            self.env = "train"
        else:
            self.env = env 
        
        self.return_index = return_index
        self.num_classes = 6

        self.samples = {}
        self.num_samples = 0

        self.samples_folder = path.join(self.data_dir, self.env)

        for i, file in enumerate(sorted(os.listdir(self.samples_folder))):
            self.samples[i] = {
                "image_path": path.join(self.samples_folder, file),
                "class_label": int(BAR.classes_str_to_idx[file.split("_")[0]]),
                "bias_label": 1
            }
            self.num_samples += 1

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]
        
        image = self.transform(Image.open(self.samples[index]["image_path"]).convert("RGB"))
        class_label = self.samples[index]["class_label"]
        bias_label = self.samples[index]["bias_label"]

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
        return f"BAR(env={self.env}, bias_amount=Unknown, num_classes={self.num_classes})"