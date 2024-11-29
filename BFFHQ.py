#!/usr/bin/env

import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator, Union

class BFFHQ(Dataset):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def __init__(self, data_dir="./data/bffhq", env="train", bias_amount=0.995, return_index = False):
        self.data_dir = data_dir
        self.transform = BFFHQ.train_transform if env == "train" else BFFHQ.eval_transform
        self.env = env
        self.bias_amount=bias_amount
        self.return_index = return_index
        self.num_classes = 2

        self.bias_folder_dict = {
            0.995: "0.5pct"
        }

        if self.env == "train":
            self.samples, self.class_labels, self.bias_labels = self.load_train_samples()

        if self.env == "val":
            self.samples, self.class_labels, self.bias_labels = self.load_val_samples()

        if self.env == "test":
            self.samples, self.class_labels, self.bias_labels = self.load_test_samples()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        class_label = self.class_labels[idx]
        bias_label = self.bias_labels[idx]

        image = self.transform(Image.open(file_path))

        if self.return_index:
            return image, class_label, bias_label, idx
        
        return image, class_label, bias_label            

    def load_train_samples(self):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]
        
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "align", class_folder))):
                samples_path.append(os.path.join(self.data_dir, bias_folder, "align", class_folder, filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        for class_folder in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "conflict"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "conflict", class_folder))):
                samples_path.append(os.path.join(self.data_dir, bias_folder, "conflict", class_folder, filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))     

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def load_val_samples(self):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]

        for filename in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "valid"))):
            samples_path.append(os.path.join(self.data_dir, bias_folder, "valid", filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def load_test_samples(self):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        for filename in sorted(os.listdir(os.path.join(self.data_dir, "test"))):
                samples_path.append(os.path.join(self.data_dir, "test", filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def assign_bias_label(self, filename: str) -> int:
        no_extension = filename.split(".")[0]
        _, y, z = no_extension.split("_")
        y = int(y)
        z = int(z)
        return 1 if y == z else -1
    
    def assign_class_label(self, filename: str):
        no_extension = filename.split(".")[0]
        _, y, _ = no_extension.split("_")
        return int(y)
    
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
        return f"BFFHQ(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.num_classes})"