#!/usr/bin/env python3

from typing import Iterable, List, Literal, Set, Union

from sklearn.svm import OneClassSVM
from BAR import BAR
from BFFHQ import BFFHQ
from Waterbirds import Waterbirds
from CIFAR10C import CIFAR10C
from ResNet import ResNet
from adecs import configure_adecs, view_with_pca
from training_utils import GCELoss
from matplotlib import pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, TensorDataset, StackDataset, WeightedRandomSampler, random_split

epochs_steps_debias = 100

datasets_configs = {

    "cifar10c": lambda bias_amount: {
        "dataset_constructor": CIFAR10C,
        "dataset_kwargs": {"bias_amount": bias_amount},
        "prepare_sampler_replacement": False,
        "erm_sampler_replacement": False,
        "prepare_lr": 0.001,
        "erm_lr": 0.001,
        "debias_lr": 0.00001,
        "model_base_name": f"ResNet_cifar10c_{int(1000*bias_amount)}",
        "model_constructor": ResNet,
        "model_kwargs_prepare": {
            "embedding_dim": 128, 
            "pretrained": False, 
            "target_image_width": 32, 
            "arch_id": 18,
            "num_classes": 10
        },
        "model_kwargs_erm": {
            "embedding_dim": 128, 
            "pretrained": False, 
            "target_image_width": 32, 
            "arch_id": 18,
            "num_classes": 10
        },
        "model_kwargs_debias": {
            "embedding_dim": 128, 
            "pretrained": False, 
            "target_image_width": 32, 
            "arch_id": 18,
            "num_classes": 10
        }
    },

    "waterbirds": lambda _ : { # Bias-Amount not configurable
        "dataset_constructor": Waterbirds,
        "dataset_kwargs": {},
        "prepare_sampler_replacement": True,
        "erm_sampler_replacement": True, 
        "prepare_lr": 0.00001,#0.000001,
        "erm_lr": 0.00001,
        "debias_lr": 0.00001,
        "model_base_name": "ResNet_Waterbirds",
        "model_constructor": ResNet,
        "model_kwargs_prepare": {
            "embedding_dim": 128, 
            "pretrained": False,
            "target_image_width": 224,
            "arch_id": 50,
            "num_classes": 2
        },
        "model_kwargs_erm": {
            "embedding_dim": 128, 
            "pretrained": True,
            "target_image_width": 224,
            "arch_id": 50,
            "num_classes": 2
        },
        "model_kwargs_debias": {
            "embedding_dim": 128, 
            "pretrained": True,
            "target_image_width": 224,
            "arch_id": 50,
            "num_classes": 2
        },
    },

    "bar": lambda _ : { # Bias Unknown
        "dataset_constructor": BAR,
        "dataset_kwargs": {},
        "prepare_sampler_replacement": True,
        "erm_sampler_replacement": True,
        "prepare_lr": 0.00001,
        "erm_lr": 0.00001,
        "debias_lr": 0.00001,
        "model_base_name": "ResNet_BAR",
        "model_constructor": ResNet,
        "model_kwargs_prepare": {
            "embedding_dim": 128, 
            "pretrained": False, 
            "target_image_width": 224, 
            "arch_id": 18,
            "num_classes": 6
        },
        "model_kwargs_erm": {
            "embedding_dim": 128, 
            "pretrained": True, 
            "target_image_width": 224, 
            "arch_id": 18,
            "num_classes": 6
        },
        "model_kwargs_debias": {
            "embedding_dim": 128, 
            "pretrained": True, 
            "target_image_width": 224, 
            "arch_id": 18,
            "num_classes": 6
        }
    },

    "bffhq": lambda _ : {
        "dataset_constructor": BFFHQ,
        "dataset_kwargs": { "bias_amount": 0.995 },
        "prepare_sampler_replacement": False,
        "erm_sampler_replacement": False,
        "prepare_lr": 0.00001,#0.0001,
        "erm_lr": 0.001,
        "debias_lr": 0.00001,#0.000001,
        "model_base_name": "ResNet_BFFHQ",
        "model_constructor": ResNet,
        "model_kwargs_prepare": {
            "embedding_dim": 128, 
            "pretrained": False, 
            "target_image_width": 224, 
            "arch_id": 18,
            "num_classes": 2
        },
        "model_kwargs_erm": {
            "embedding_dim": 128, 
            "pretrained": True, 
            "target_image_width": 224, 
            "arch_id": 18,
            "num_classes": 2
        },
        "model_kwargs_debias": {
            "embedding_dim": 128, 
            "pretrained": True, 
            "target_image_width": 224, 
            "arch_id": 18,
            "num_classes": 2
        }
    }
}


def write_temp_tables(
        dataset: str, 
        bias_amount: float, 
        save_results_to: str, 
        validation_accuracies_avg: list, 
        validation_accuracies_b: list, 
        validation_accuracies_u: list, 
        test_accuracies_avg: list, 
        test_accuracies_b: list, 
        test_accuracies_u: list
    ) -> None:

    import numpy as np

    num_epochs = len(test_accuracies_avg)
    best_results_index = np.argmax(test_accuracies_u) if dataset == "waterbirds" else np.argmax(test_accuracies_avg)

    with open(f"{save_results_to}/TEX_{dataset}_{int(bias_amount * 1000)}.tex", mode="a+") as tex:
        tex.write(" & & Val Avg & Val (B) & Val (U) & Test Avg & Test (B) & Test (U) \\\\ \n")
        
        tex.write(
            f"Exit (Epoch: {num_epochs}/{num_epochs}) & "
            f"{validation_accuracies_avg[-1]:.4f} & {validation_accuracies_b[-1]:.4f} & {validation_accuracies_u[-1]:.4f} & "
            f"{test_accuracies_avg[-1]:.4f} & {test_accuracies_b[-1]:.4f} & {test_accuracies_u[-1]:.4f} \\\\ \n"
        )

        tex.write(
            f"Best (Epoch: {best_results_index+1}/{num_epochs}) & "
            f"{validation_accuracies_avg[best_results_index]:.4f} & {validation_accuracies_b[best_results_index]:.4f} & {validation_accuracies_u[best_results_index]:.4f} & "
            f"{test_accuracies_avg[best_results_index]:.4f} & {test_accuracies_b[best_results_index]:.4f} & {test_accuracies_u[best_results_index]:.4f} \\\\ \n"
        )

    return test_accuracies_avg[best_results_index], test_accuracies_u[best_results_index]


def pretrain_GCE(
    dataset: str, 
    bias_amount: float, 
    batch_size: int, 
    target_batch_size, 
    num_epochs: int
):
    model_constructor:      ResNet #type-annot    
    model:                  ResNet #type-annot    
    
    accumulation_steps = target_batch_size // batch_size
    
    print("BIAS_AMOUNT", bias_amount)
    config = datasets_configs[dataset](bias_amount)

    learning_rate_step_prepare = config["prepare_lr"]
    model_name = config["model_base_name"]
    prepare_model_name = f"{model_name}_step_0.pt"
    
    dataset_constructor = config["dataset_constructor"]
    dataset_kwargs = config["dataset_kwargs"]
    prepare_sampler_replacement = config["prepare_sampler_replacement"]

    model_constructor = config["model_constructor"]
    model_kwargs = config["model_kwargs_prepare"]
        
    train_set = dataset_constructor(
        env="train",
        return_index=True,
        **dataset_kwargs
    )
    val_set = dataset_constructor(
        env="val",
        **dataset_kwargs
    )
    test_set = dataset_constructor(
        env="test",
        **dataset_kwargs
    )

    print("DATASETS:")
    print(train_set)
    print(val_set)
    print(test_set)
    
    class_pops, class_labels = train_set.perclass_populations(return_labels=True)
    class_weights: torch.Tensor = torch.as_tensor([1.0 / class_pops[y] for y in class_labels])
    classweighted_sampler: WeightedRandomSampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(train_set),
        replacement=prepare_sampler_replacement
    )

    print("\nSizes of Sets:")
    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=classweighted_sampler,  pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    model = model_constructor(
        prepare_erm_loss = True,
        **model_kwargs
    )
    
    print(f"Using {model.erm_loss_fn}")
    validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u, test_accuracies_avg, test_accuracies_b, test_accuracies_u = model.train_model_erm(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=learning_rate_step_prepare,
        num_epochs=num_epochs,
        accumulate=accumulation_steps        
    )
    
    model_dir = f"./PreparedModels/{num_epochs}"
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/{prepare_model_name}")
    print("Saved GCE Model:")
    print(f"\t -- Dataset: {dataset}")
    print(f"\t -- Bias Amount: {bias_amount}")
    print(f"\t -- Epochs: {num_epochs}")
    print(f"\t -- Model Path: {model_dir}/{prepare_model_name}")

def pretrain_ERM(
    dataset: str,
    bias_amount: float,
    batch_size: int,
    target_batch_size: int,
    num_epochs: int
) -> None:
    
    model_constructor: ResNet #type-annot
    model:             ResNet #type-annot

    accumulation_steps = target_batch_size // batch_size

    print("BIAS_AMOUNT", bias_amount)
    config = datasets_configs[dataset](bias_amount)

    learning_rate_step_erm = config["erm_lr"]
    model_name = config["model_base_name"]
    erm_model_name = f"{model_name}_erm_{num_epochs}.pt"

    dataset_constructor = config["dataset_constructor"]
    dataset_kwargs = config["dataset_kwargs"]
    erm_sampler_replacement = config["erm_sampler_replacement"]
    
    model_constructor = config["model_constructor"]
    model_kwargs      = config["model_kwargs_erm"]

    train_set = dataset_constructor(
        env="train",
        return_index=True,
        **dataset_kwargs
    )

    val_set = dataset_constructor(
        env="val",
        **dataset_kwargs
    )

    test_set = dataset_constructor(
        env="test",
        **dataset_kwargs
    )

    print("DATASETS:")
    print(train_set)
    print(val_set)
    print(test_set)

    class_pops, class_labels = train_set.perclass_populations(return_labels=True)
    class_weights: torch.Tensor = torch.as_tensor([1.0 / class_pops[y] for y in class_labels])
    classweighted_sampler: WeightedRandomSampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(train_set),
        replacement=erm_sampler_replacement
    )

    print("\nSizes of Sets:")
    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    print(torch.as_tensor(list(train_set.get_bias_labels())).unique(return_counts=True))

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=classweighted_sampler,  pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    model = model_constructor(
        prepare_erm_loss = False,
        **model_kwargs
    )
    
    print(f"Using {model.erm_loss_fn}")
    validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u, test_accuracies_avg, test_accuracies_b, test_accuracies_u = model.train_model_erm(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=learning_rate_step_erm,
        num_epochs=num_epochs,
        accumulate=accumulation_steps        
    )
    
    model_dir = f"./PreparedModels/ERM"
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/{erm_model_name}")
    print("Saved ERM Model:")
    print(f"\t -- Dataset: {dataset}")
    print(f"\t -- Bias Amount: {bias_amount}")
    print(f"\t -- Epochs: {num_epochs}")
    print(f"\t -- Model Path: {model_dir}/{erm_model_name}")
    
    os.makedirs(f"{model_dir}/features_viz/{num_epochs}", exist_ok=True)
    tr_features, tr_labels, tr_bias_labels = model.extract_features(train_loader)
    view_with_pca(tr_features, tr_labels, tr_bias_labels, num_classes=train_set.num_classes, save_results_to=f"{model_dir}/features_viz/{num_epochs}")
    write_temp_tables(
        dataset=dataset,
        bias_amount=bias_amount,
        save_results_to=model_dir,
        validation_accuracies_avg=validation_accuracies_avg,
        validation_accuracies_b=validation_accuracies_b,
        validation_accuracies_u=validation_accuracies_u,
        test_accuracies_avg=test_accuracies_avg,
        test_accuracies_b=test_accuracies_b,
        test_accuracies_u=test_accuracies_u
    )

def debias_from_adecs(
    dataset: str, 
    bias_amount: float, 
    batch_size: int = 256,
    target_batch_size: int = 256,
    step_0_from_num_epochs: int = 200, 
    erm_from_num_epochs: int = 100,
    contamination: float = 0.5,
    gamma: Union[float, Literal["scale"], Literal["auto"]] = "scale",
    error_ratio: float = 0.5,
    normalize: bool = True,
    custom_save_results_to = None
):
    model_constructor:      ResNet #type-annot    
    model:                  ResNet #type-annot    

    print("BIAS_AMOUNT", bias_amount)
    config = datasets_configs[dataset](bias_amount)
    model_name = config["model_base_name"]
    prepare_model_name = f"{model_name}_step_0.pt"
    erm_model_name = f"{model_name}_erm_{erm_from_num_epochs}.pt"
    dataset_constructor = config["dataset_constructor"]
    dataset_kwargs = config["dataset_kwargs"]
    model_constructor = config["model_constructor"]
    model_kwargs = config["model_kwargs_prepare"]

    train_set = dataset_constructor(
        env="train",
        return_index=True,
        **dataset_kwargs
    )

    val_set = dataset_constructor(
        env="val",
        **dataset_kwargs
    )

    test_set = dataset_constructor(
        env="test",
        **dataset_kwargs
    )

    print("\nSizes of Sets:")
    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    print(f"Loading ./PreparedModels/ERM/{erm_model_name}")
    model = model_constructor.load_model(f"./PreparedModels/ERM/{erm_model_name}", **model_kwargs)
    model.test_model(test_loader)

    print(f"Loading ./PreparedModels/{step_0_from_num_epochs}/{prepare_model_name}")
    model = model_constructor.load_model(
        filepath=f"./PreparedModels/{step_0_from_num_epochs}/{prepare_model_name}",
        **model_kwargs
    )
    
    train_loader: DataLoader    = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    save_results_to = f"./DebiasAdecs_classauto/{dataset}/{str(bias_amount).replace('.', '')}/_{model_name}_{int(1000 * error_ratio)}" if custom_save_results_to is None else \
        custom_save_results_to
    os.makedirs(save_results_to, exist_ok=True)

    _, _, total_per_class, misclassified_per_class = model.misclassified_statistics(train_loader, save_results_to)

    percentile = (((misclassified_per_class / total_per_class) * error_ratio) * 100).numpy()
    
    _, _, _, _, adecs_preds, _ = configure_adecs(
        model=model,
        train_loader=train_loader,
        num_classes=train_set.num_classes,
        percentile=percentile,
        contamination=[contamination for _ in range(train_set.num_classes)],
        gamma="scale",
        bias_amount=bias_amount,
        save_results_to=save_results_to,
        make_figures=True,
        normalization=True
    )
    
    data_augmentation = True
    ground_truth = False
    mixup = False
    
    adecs_preds_set: TensorDataset = TensorDataset(adecs_preds)
    adecs_labels:    torch.Tensor = torch.as_tensor(list(train_set.get_bias_labels()))
    stacked_train_set: StackDataset  = StackDataset(train_set, adecs_preds_set)
    
    bias_unique_labels: torch.Tensor #type-annot
    counts: torch.Tensor #type-annot
    
    bias_unique_labels, counts = adecs_preds.unique(return_counts=True) if ground_truth is False \
        else adecs_labels.unique(return_counts=True)    
    
    sampling_weights: torch.Tensor  = torch.as_tensor([1.0 / counts[p] for p in ((adecs_preds + 1) * 0.5).long()]) if ground_truth is False \
        else torch.as_tensor([1.0 / counts[p] for p in ((adecs_labels.float() + 1) * 0.5).long()])
    
    print(bias_unique_labels, counts)
    print(sampling_weights.unique(return_counts=True))
    
    bias_weighted_sampler: WeightedRandomSampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(stacked_train_set, batch_size=batch_size, sampler=bias_weighted_sampler)

    accumulation_steps = target_batch_size // batch_size
    learning_rate_debias = config["debias_lr"]
    
    debiased_model_name = f"{model_name}_debiased_adecs.pt"

    model = model_constructor(
        prepare_erm_loss = False,
        **model_kwargs
    )

    print(f"Loading ./PreparedModels/ERM/{erm_model_name}")
    model = model_constructor.load_model(f"./PreparedModels/ERM/{erm_model_name}", **model_kwargs)
    try:
        validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u, test_accuracies_avg, test_accuracies_b, test_accuracies_u = model.train_model_werm_2(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            ground_truth=ground_truth,
            data_augmentation=data_augmentation,
            mixup=mixup,
            learning_rate=learning_rate_debias,
            num_epochs=epochs_steps_debias,
            accumulate=accumulation_steps
        )
    except KeyboardInterrupt:
        print("Debiasing interrupted, saving model.")
        pass

    best_avg_acc, best_unb_acc = write_temp_tables(
        dataset=dataset,
        bias_amount=bias_amount,
        save_results_to=save_results_to,
        validation_accuracies_avg=validation_accuracies_avg,
        validation_accuracies_b=validation_accuracies_b,
        validation_accuracies_u=validation_accuracies_u,
        test_accuracies_avg=test_accuracies_avg,
        test_accuracies_b=test_accuracies_b,
        test_accuracies_u=test_accuracies_u
    )
    
    model.save_model(debiased_model_name)

    return best_avg_acc, best_unb_acc

if __name__ == "__main__":    
    epochs_steps_debias = 100
    num_epochs = 100
    for _ in range(3):        
        for bias_amount in [0.95, 0.98, 0.99, 0.995]:
            pretrain_GCE("cifar10c", bias_amount=bias_amount, batch_size=256, target_batch_size=256, num_epochs=num_epochs)
            pretrain_ERM("cifar10c", bias_amount=bias_amount, batch_size=256, target_batch_size=256, num_epochs=num_epochs)
            debias_from_adecs("cifar10c", bias_amount=bias_amount, batch_size=256, target_batch_size=256, step_0_from_num_epochs=num_epochs, erm_from_num_epochs=num_epochs)
    




    
