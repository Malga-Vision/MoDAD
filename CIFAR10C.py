# coding: utf-8

import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator, Union
from torch.utils.data import ConcatDataset

class CIFAR10C(Dataset):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data_dir="./data/cifar10c", env="train", bias_amount=0.95, return_index = False):
        self.data_dir = data_dir
        self.transform = CIFAR10C.train_transform if env == "train" else CIFAR10C.eval_transform
        self.env=env
        self.bias_amount=bias_amount
        self.return_index = return_index
        self.num_classes = 10

        self.bias_folder_dict = {
            0.95: "5pct",
            0.98: "2pct",
            0.99: "1pct",
            0.995: "0.5pct",

        }
        if self.env == "train":
            self.samples, self.class_labels, self.bias_labels = self.load_train_samples()

        if self.env == "val":
            self.samples, self.class_labels, self.bias_labels = self.load_val_samples()

        if self.env == "test":
            self.samples, self.class_labels, self.bias_labels = self.load_test_samples()

    def load_train_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]
        bias_folder=self.bias_folder_dict[self.bias_amount]
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "align",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "align",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))
        
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "conflict"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "conflict",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "conflict",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))
        
        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 
    
    def load_val_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        bias_folder=self.bias_folder_dict[self.bias_amount]
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "valid"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder,"valid",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "valid",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 
    
    def load_test_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,"test"))):
            for filename in  sorted(os.listdir(os.path.join(self.data_dir,"test",class_folder))):
                samples_path.append(os.path.join(self.data_dir,"test",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        class_label=self.class_labels[idx]
        bias_label=self.bias_labels[idx]

        image = self.transform(Image.open(file_path))   #senza self.transofrm per vedere le immagini 
        
        if self.return_index:
            return image, class_label, bias_label, idx

        return image, class_label, bias_label

    def assign_bias_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, z = no_extension.split('_')
        y, z = int(y), int(z)
        if y == z:
            return 1
        return -1
    
    def assign_class_label(self, filename):
        no_extension=filename.split('.')[0]
        
        _, y, _ = no_extension.split('_')
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
        return f"CIFAR10C(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.num_classes})"
