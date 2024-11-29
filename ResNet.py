from typing import Iterable, List, Tuple, Union
from sklearn.svm import OneClassSVM
from torchvision.models import ResNet50_Weights, resnet50, ResNet18_Weights, resnet18, resnet152, ResNet152_Weights
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from torch import nn
from torch import optim
from training_utils import LRDecayWithPatience, EarlyStopping, GCELoss
import numpy as np
import gc
from sklearn.metrics import classification_report, confusion_matrix

RESNET50_BACKBONE_MODULES_PRETRAINED = lambda : [module for _, module in resnet50(weights=ResNet50_Weights.DEFAULT).named_children()][:-1]
RESNET18_BACKBONE_MODULES_PRETRAINED = lambda : [module for _, module in resnet18(weights=ResNet18_Weights.DEFAULT).named_children()][:-1]
RESNET50_BACKBONE_MODULES = lambda : [module for _, module in resnet50(weights=None).named_children()][:-1]
RESNET18_BACKBONE_MODULES = lambda : [module for _, module in resnet18(weights=None).named_children()][:-1]


class ResNet(nn.Module):
    def __init__(self, arch_id: int = 50, pretrained: bool = False, embedding_dim = None, num_classes = 10, target_image_width: int = 224, prepare_erm_loss=False, *args, **kwargs) -> None:
        
        if arch_id not in {18, 50}:
            raise ValueError("Valid 'arch_id' parameter values are either '50' or '18', as integers")
        
        self.backbone_head_link_neurons = 2048 if arch_id == 50 else 512
        
        super().__init__(*args, **kwargs)

        self.embedding_dim = embedding_dim

        resnet_backbone_modules: list[nn.Module] = []
        self.num_classes: int = num_classes
        self.target_image_width: int = target_image_width

        match pretrained:
            case True:
                resnet_backbone_modules = RESNET50_BACKBONE_MODULES_PRETRAINED() if arch_id == 50 else RESNET18_BACKBONE_MODULES_PRETRAINED()
            case False:
                resnet_backbone_modules = RESNET50_BACKBONE_MODULES() if arch_id == 50 else RESNET18_BACKBONE_MODULES()        
        
        head_layers_link = [nn.Flatten(), nn.ReLU()] + \
            ([] if embedding_dim is None else [nn.Linear(self.backbone_head_link_neurons, embedding_dim), nn.ReLU()])
        
        self.backbone_nn: nn.Sequential = nn.Sequential(
            *resnet_backbone_modules, 
            *head_layers_link
        )
        
        self.linear_head_nn: nn.Linear = nn.Linear(self.backbone_head_link_neurons, num_classes) if embedding_dim is None \
            else nn.Linear(embedding_dim, num_classes)

        self.model: nn.Sequential = nn.Sequential(
            self.backbone_nn,
            self.linear_head_nn
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.erm_loss_fn  = nn.CrossEntropyLoss(reduction="mean") if prepare_erm_loss is False else GCELoss(q=0.7, reduction="mean") 
        self.werm_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.mix_loss_fn  = nn.MSELoss(reduction="mean") 
        
        self.transforms_set1 = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.Resize(size=(self.target_image_width, self.target_image_width), antialias=True),
        ])

        # Second set of transforms
        self.transforms_set2 = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomAutocontrast(p=0.5),  
            transforms.Resize(size=(self.target_image_width, self.target_image_width), antialias=True),
        ])

        # Third set of transforms
        self.transforms_set3 = transforms.Compose([
            transforms.CenterCrop((self.target_image_width // 2, self.target_image_width // 2)),
            transforms.Resize(size=(self.target_image_width, self.target_image_width), antialias=True),
        ])  

        self.to(self.device)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_nn(x)
    
    def head(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_head_nn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def misclassified_statistics(self, dataloader: DataLoader, save_results_to: Union[str, None] = None) -> TensorDataset:        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        correct_biased = 0
        correct_unbiased = 0
        total_biased = 0
        total_unbiased = 0
        correct_per_class = torch.zeros(self.num_classes)
        total_per_class   = torch.zeros(self.num_classes)
        
        perclass_bias_preds:  list[list] = [list() for _ in range(self.num_classes)]
        perclass_bias_labels: list[list] = [list() for _ in range(self.num_classes)]
        bias_preds:           list = []       
        bias_targs:           list = []      

        logits = []
        rel_blabels = []

        with torch.no_grad():
            for inputs, labels, bias_labels, _ in dataloader:
                inputs, labels, bias_labels = self.put_on_device(inputs, labels, bias_labels)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.erm_loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                bias_preds.append(torch.where(predicted == labels, 1, -1))
                bias_targs.append(bias_labels)

                logits.append(self(inputs[predicted != labels]))
                rel_blabels.append(bias_labels[predicted != labels])
                
                for _class in torch.arange(self.num_classes):
                    _class = _class.to(self.device)
                    inclass_labels: torch.Tensor = labels[labels == _class].to(self.device)
                    inclass_preds: torch.Tensor  = predicted[labels == _class].to(self.device)
                    total_per_class[_class] += inclass_labels.size(0)
                    correct_per_class[_class] += (inclass_preds == inclass_labels).sum().item()
                    perclass_bias_labels[_class].append(bias_labels[labels == _class])
                    perclass_bias_preds[_class].append(torch.where(inclass_preds == inclass_labels, 1, -1))
                 
                bias_predicted = predicted[bias_labels != -1]
                total_biased += len(bias_predicted)
                correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                unbias_predicted = predicted[bias_labels == -1]
                total_unbiased += len(unbias_predicted)
                correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            num_misclassified_samples: int = total - correct
            misclassified_per_class = total_per_class - correct_per_class
            accuracy_biased = 100 * correct_biased / total_biased
            accuracy_unbiased = 100 * correct_unbiased / (total_unbiased+0.0000001)            
        
        print(f"\t -- Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"\t\t Accuracy Biased: {accuracy_biased:.4f}%, Accuracy Unbiased: {accuracy_unbiased:.4f}%")
        print(f"# Misclassified Samples: {num_misclassified_samples}")
        print(f"# Misclassified per class:", misclassified_per_class)

        logits = torch.cat(logits, dim=0).cpu()
        rel_blabels = torch.cat(rel_blabels, dim=0).cpu()

        perclass_bias_preds  = [torch.cat(p, dim=0).cpu() for p in perclass_bias_preds]
        perclass_bias_labels = [torch.cat(p, dim=0).cpu() for p in perclass_bias_labels]

        # for _class in torch.arange(self.num_classes):
        #     if save_results_to:
        #         with open(f"{save_results_to}/mistakes_preds.txt", mode="a+") as f:
        #             f.write(f"Class {_class}\n")
        #             try:
        #                 f.write(classification_report(perclass_bias_labels[_class], perclass_bias_preds[_class], target_names=["Unbiased", "Biased"]))
        #             except ValueError:
        #                 print("Error during classification report")
        #                 pass
        #     else:
        #         print(f"Class {_class}\n")
        #         try:
        #             print(classification_report(perclass_bias_labels[_class], perclass_bias_preds[_class], target_names=["Unbiased", "Biased"]))
        #         except ValueError:
        #             print("Error during classification report")
        #             pass
        # print("Mistakes CF:")
        # print(confusion_matrix(torch.cat(bias_targs, dim=0).cpu().numpy(), torch.cat(bias_preds, dim=0).cpu().numpy()))

        return torch.cat(bias_preds, dim=0).cpu(), num_misclassified_samples, total_per_class, misclassified_per_class

    def test_model(self, test_loader):
        self.eval()
        total_loss_test = 0.0
        correct_test = 0
        total_test = 0
        correct_biased_test = 0
        correct_unbiased_test = 0
        total_biased_test = 0
        total_unbiased_test = 0

        with torch.no_grad():
            for inputs, labels, bias_labels in test_loader:
                inputs, labels, bias_labels = self.put_on_device(inputs, labels, bias_labels)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.erm_loss_fn(outputs, labels)
                total_loss_test += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                bias_predicted = predicted[bias_labels != -1]
                total_biased_test += len(bias_predicted)
                correct_biased_test += (bias_predicted == labels[bias_labels != -1]).sum().item()

                unbias_predicted = predicted[bias_labels == -1]
                total_unbiased_test += len(unbias_predicted)
                correct_unbiased_test += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_test = total_loss_test / len(test_loader)
            accuracy_test = 100 * correct_test / total_test

            accuracy_test_biased = 100 * correct_biased_test / total_biased_test
            accuracy_test_unbiased = 100 * correct_unbiased_test / (total_unbiased_test+0.0000001)            
        
        print(f"\t -- Test Loss: {average_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%")
        print(f"\t\t Test Accuracy Biased: {accuracy_test_biased:.4f}%, Test Accuracy Unbiased: {accuracy_test_unbiased:.4f}%")
    
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        features: list = []
        labels:   list = []
        bias_labels: list = []

        self.eval()
        with torch.no_grad():
            try:
                for _inputs, _labels, _bias_labels in dataloader:
                    _inputs: torch.Tensor = _inputs.to(self.device)
                    _labels: torch.Tensor = _labels.to(self.device)                
                    _bias_labels: torch.Tensor = _bias_labels.to(self.device)
                    fx = self.backbone(_inputs)
                    features.append(fx.squeeze(-2, -1))
                    labels.append(_labels)
                    bias_labels.append(_bias_labels)
            
            except ValueError: # 'Too many values to unpack... ' when dataset's return_index is True
                for _inputs, _labels, _bias_labels, _ in dataloader:
                    _inputs: torch.Tensor = _inputs.to(self.device)
                    _labels: torch.Tensor = _labels.to(self.device)                
                    _bias_labels: torch.Tensor = _bias_labels.to(self.device)
                    fx = self.backbone(_inputs)
                    features.append(fx.squeeze(-2, -1))
                    labels.append(_labels)
                    bias_labels.append(_bias_labels)

        return (
            torch.cat(features, dim=0).cpu().numpy(), 
            torch.cat(labels, dim=0).cpu().numpy(), 
            torch.cat(bias_labels, dim=0).cpu().numpy()
        )
    
    def freeze_backbone(self):
        self.backbone_nn.requires_grad_(False)
        self.backbone_nn.eval()

    def unfreeze_backbone(self):
        self.backbone_nn.requires_grad_(True)
        self.backbone_nn.train()

    def put_on_device(self, *tensors: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        return (tensor.to(self.device) for tensor in tensors)
    
    def train_model_erm(
            self,
            train_loader,
            val_loader,
            test_loader,
            learning_rate=0.001,
            num_epochs=50,
            accumulate=1):
        
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, amsgrad=False)

        validation_accuracies_avg = []
        validation_accuracies_b   = []
        validation_accuracies_u   = []

        test_accuracies_avg       = []
        test_accuracies_b         = []
        test_accuracies_u         = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_train = 0
            total_train = 0
            total_biased = 0
            correct_biased = 0
            total_unbiased = 0
            correct_unbiased = 0
            
            self.train()
            optimizer.zero_grad()
            with torch.enable_grad():
                for batch_idx, (inputs, labels, bias_labels, _) in enumerate(train_loader):
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)

                    outputs: torch.Tensor = self(inputs)                    

                    loss: torch.Tensor = self.erm_loss_fn(outputs, labels) / accumulate
                    loss.backward(retain_graph=True)                    
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

                    # self.eval()
                    # gradients: torch.Tensor = per_sample_gradient(self, self.erm_loss_fn, inputs, labels)
                    # self.train()

                    if ((batch_idx + 1) % accumulate == 0) or (batch_idx + 1 == len(train_loader)):                         
                        optimizer.step()
                        optimizer.zero_grad()

                
            average_loss_train = total_loss / len(train_loader)
            accuracy_train = 100 * correct_train / total_train
            accuracy_train_biased = 100 * correct_biased/total_biased
            accuracy_train_unbiased = 100 * correct_unbiased/(total_unbiased+0.000001)
            tr_tot_unbiased = total_unbiased

            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            correct_biased = 0
            correct_unbiased = 0
            total_biased = 0
            total_unbiased = 0
            
            
            with torch.no_grad():
                for inputs, labels, bias_labels in val_loader:
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)

                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_val = val_loss / len(val_loader)
            accuracy_val_biased = 100*correct_biased/total_biased
            accuracy_val_unbiased = 100*correct_unbiased/(total_unbiased+0.000001)
            accuracy_val = 100 * correct_val / total_val
            val_tot_unbiased = total_unbiased

            validation_accuracies_avg.append(accuracy_val)
            validation_accuracies_b.append(accuracy_val_biased)
            validation_accuracies_u.append(accuracy_val_unbiased)

            self.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            correct_biased=0
            correct_unbiased=0
            total_biased=0
            total_unbiased=0

            with torch.no_grad():
                for inputs, labels, bias_labels in test_loader:
                    bias_labels = bias_labels.to(self.device)
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)

                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_test = test_loss / len(test_loader)
            accuracy_test = 100 * correct / total
            accuracy_test_biased = 100 * correct_biased / total_biased
            accuracy_test_unbiased = 100 * correct_unbiased / (total_unbiased+0.000001) 
            te_tot_unbiased = total_unbiased

            test_accuracies_avg.append(accuracy_test)
            test_accuracies_b.append(accuracy_test_biased)
            test_accuracies_u.append(accuracy_test_unbiased)   
                     
            
            print(f"Epoch {epoch + 1}/{num_epochs}\n\t -- Loss: {average_loss_train:.4f}, Train Accuracy: {accuracy_train:.2f}%")
            print(f"\t\t Train Accuracy Biased: {accuracy_train_biased:.4f} %, Train Accuracy Unbiased: {accuracy_train_unbiased:.4f}% ({tr_tot_unbiased})")
            print(f"\t -- Validation Loss: {average_loss_val:.4f}, Validation Accuracy: {accuracy_val:.4f} %")
            print(f"\t\t Valid Accuracy Biased: {accuracy_val_biased:.4f} %, Valid Accuracy Unbiased: {accuracy_val_unbiased:.4f}% ({val_tot_unbiased})")
            print(f"\t -- Test Loss: {average_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%")
            print(f"\t\t Test Accuracy Biased: {accuracy_test_biased:.4f}%, Test Accuracy Unbiased: {accuracy_test_unbiased:.4f}% ({te_tot_unbiased})") 
        
        return (
            validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u, 
            test_accuracies_avg, test_accuracies_b, test_accuracies_u
        )

    def train_model_werm_2(
            self,
            train_loader,
            val_loader,
            test_loader,
            ground_truth,
            data_augmentation,
            mixup,
            learning_rate=0.0001,
            num_epochs=30,
            accumulate=1
        ):        
                                                                                                          
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, amsgrad=False)
        early_stopping = EarlyStopping(warmup=300, patience=25, epsilon=1e-6)
        
        validation_accuracies_avg = []
        validation_accuracies_b   = []
        validation_accuracies_u   = []

        test_accuracies_avg       = []
        test_accuracies_b         = []
        test_accuracies_u         = []

        for epoch in range(num_epochs):
            total_loss_train = 0.0
            correct_train = 0
            total_train = 0
            correct_biased_train = 0
            correct_unbiased_train = 0
            total_biased_train = 0
            total_unbiased_train = 0         
            
            true_samples = 0
            augmented_samples = 0

            total_counts_bu = torch.Tensor([0, 0]).to(self.device)
            
            self.train()
            optimizer.zero_grad()
            with torch.enable_grad():
                for batch_idx, ((inputs, labels, bias_labels, _), (adecs_preds, )) in enumerate(train_loader): 
                    inputs, labels, bias_labels, adecs_preds = self.put_on_device(inputs, labels, bias_labels, adecs_preds)
                    blabel_set, counts = torch.unique(bias_labels, return_counts=True)
                    counts = counts.to(self.device)
                    total_counts_bu += counts
                    if ground_truth:
                        adecs_preds = bias_labels
                    labels = labels.type(torch.LongTensor)

                    unbias_weight = 1 - (torch.count_nonzero(adecs_preds == 1) / len(adecs_preds))
                    weights_tensor: torch.Tensor = torch.where(adecs_preds == -1, unbias_weight, 1 - unbias_weight).to(self.device)                
                    
                    labels = torch.as_tensor(labels).to(self.device)
                    true_samples += len(labels)

                    # inputs = self.transforms_set2(inputs)

                    if data_augmentation == True:
                        if torch.sum(adecs_preds == -1) > 0:
                            
                            samples_to_augment: torch.Tensor = inputs[adecs_preds == -1] 
                            unb_samp_1 = self.transforms_set1(samples_to_augment)
                            unb_samp_2 = self.transforms_set2(samples_to_augment)
                            unb_samp_3 = self.transforms_set3(samples_to_augment)

                            unb_samples = torch.cat([unb_samp_1, unb_samp_2, unb_samp_3], dim=0) # unb_samp_2, unb_samp_3
                            total_counts_bu += torch.Tensor([unb_samples.size(0), 0.]).to(self.device)                            
                            inputs = torch.cat([unb_samples, inputs], dim=0) 
                            
                            bias_labels_uns = bias_labels[adecs_preds == -1]
                            bias_labels = torch.cat([bias_labels_uns, bias_labels_uns, bias_labels_uns, bias_labels], dim=0)

                            labels_unb = labels[adecs_preds == -1]                                             
                            labels = torch.cat([labels_unb, labels_unb, labels_unb, labels], dim=0)

                            weights_uns = weights_tensor[adecs_preds == -1]
                            weights_tensor = torch.cat([weights_uns, weights_uns, weights_uns, weights_tensor], dim=0)
                            
                            adecs_preds_uns = adecs_preds[adecs_preds == -1]
                            adecs_preds = torch.cat([adecs_preds_uns, adecs_preds_uns, adecs_preds_uns, adecs_preds], dim=0)
                            augmented_samples += unb_samples.size(0)

                        shuffle_idxs: torch.Tensor = torch.randperm(labels.size(0)).to(self.device)
                        inputs = inputs[shuffle_idxs]
                        labels = labels[shuffle_idxs]
                        bias_labels = bias_labels[shuffle_idxs]
                        weights_tensor = weights_tensor[shuffle_idxs]
                        adecs_preds = adecs_preds[shuffle_idxs]

                    outputs: torch.Tensor = self(inputs)
                    loss: torch.Tensor = self.werm_loss_fn(outputs, labels)               
                    loss = (loss * weights_tensor).mean() 
                    
                    if mixup: 
                        num_mix_samples: int = min(torch.count_nonzero(adecs_preds == 1), torch.count_nonzero(adecs_preds == -1))
                        
                        if num_mix_samples > 1:                            
                            b_inputs: torch.Tensor = inputs[adecs_preds == 1][:num_mix_samples]
                            u_inputs: torch.Tensor = inputs[adecs_preds == -1][:num_mix_samples]
                            b_labels: torch.Tensor = labels[adecs_preds == 1][:num_mix_samples]             
                            u_labels: torch.Tensor = labels[adecs_preds == -1][:num_mix_samples]

                            b_labels = torch.nn.functional.one_hot(b_labels, num_classes=self.num_classes)
                            u_labels = torch.nn.functional.one_hot(u_labels, num_classes=self.num_classes)

                            alpha: torch.Tensor = torch.distributions.Uniform(0.1, 0.5).sample()
                            _lambda: torch.Tensor = torch.distributions.Beta(alpha, alpha).sample().to(self.device) # (u_labels.size(0), )
                            # _lambda = _lambda.unsqueeze(-1).unsqueeze(-1) # to make it broadcastable to batch
                                
                            mix_inputs: torch.Tensor = _lambda * b_inputs + (1 - _lambda) * u_inputs
                            mix_labels: torch.Tensor = _lambda * b_labels + (1 - _lambda) * u_labels
                            # mix_labels = mix_labels.squeeze(0)
                            
                            mix_outs: torch.Tensor = self(mix_inputs)
                            mix_loss: torch.Tensor = self.mix_loss_fn(mix_outs, mix_labels)

                            loss = (loss + 0.25*mix_loss ).mean() 

                    loss = loss / accumulate

                    loss.backward()
                    
                    total_loss_train += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased_train += len(bias_predicted)
                    correct_biased_train += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased_train += len(unbias_predicted)
                    correct_unbiased_train += (unbias_predicted == labels[bias_labels == -1]).sum().item()

                    if ((batch_idx + 1) % accumulate == 0) or (batch_idx + 1 == len(train_loader)): 
                        #updates the model's parameters using the optimizer, applying gradient descent.
                        optimizer.step()
                        optimizer.zero_grad()
                
            
            #average training loss for that epoch
            average_loss_train = total_loss_train / len(train_loader)
            accuracy_train = 100 * correct_train / total_train
            # accuracy_adecs = 100 * adecs_correct_preds / total_train

            accuracy_train_biased = 100 * correct_biased_train / total_biased_train
            accuracy_train_unbiased = 100 * correct_unbiased_train / (total_unbiased_train+0.000001)

            self.eval()
            total_loss_val = 0.0
            correct_val = 0
            total_val = 0
            correct_biased_val=0
            correct_unbiased_val=0
            total_biased_val=0
            total_unbiased_val=0
        
            #computes the validation and test losses and calculates the accuracy of the model's predictions.
            with torch.no_grad():
                for inputs, labels, bias_labels in val_loader:
                    inputs, labels, bias_labels = self.put_on_device(inputs, labels, bias_labels)
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(self.device)

                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels)
                    total_loss_val += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased_val += len(bias_predicted)
                    correct_biased_val += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased_val += len(unbias_predicted)
                    correct_unbiased_val += (unbias_predicted == labels[bias_labels == -1]).sum().item()


            average_loss_val = total_loss_val / len(val_loader)
            accuracy_val_biased = 100 * correct_biased_val / total_biased_val
            accuracy_val_unbiased = 100 * correct_unbiased_val / (total_unbiased_val+0.000001)
            # scheduler(average_loss_val)
            if early_stopping(average_loss_val, epoch):
                print("EARLY STOPPING: no improvement, stopping")
                break
            accuracy_val = 100 * correct_val / total_val

            validation_accuracies_avg.append(accuracy_val)
            validation_accuracies_b.append(accuracy_val_biased)
            validation_accuracies_u.append(accuracy_val_unbiased)

            self.eval()
            total_loss_test = 0.0
            correct_test = 0
            total_test = 0
            correct_biased_test = 0
            correct_unbiased_test = 0
            total_biased_test = 0
            total_unbiased_test = 0

            with torch.no_grad():
                for inputs, labels, bias_labels in test_loader:
                    inputs, labels, bias_labels = self.put_on_device(inputs, labels, bias_labels)
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(self.device)

                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels)
                    total_loss_test += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                    
                    bias_predicted = predicted[bias_labels != -1]
                    total_biased_test += len(bias_predicted)
                    correct_biased_test += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased_test += len(unbias_predicted)
                    correct_unbiased_test += (unbias_predicted == labels[bias_labels == -1]).sum().item()

                average_loss_test = total_loss_test / len(test_loader)
                accuracy_test = 100 * correct_test / total_test

                accuracy_test_biased = 100 * correct_biased_test / total_biased_test
                accuracy_test_unbiased = 100 * correct_unbiased_test / (total_unbiased_test+0.0000001)           

            test_accuracies_avg.append(accuracy_test)
            test_accuracies_b.append(accuracy_test_biased)
            test_accuracies_u.append(accuracy_test_unbiased)  

            print(f"Epoch {epoch + 1}/{num_epochs} ({blabel_set}, {total_counts_bu})")
            print(f"\n\t -- Train Loss: {average_loss_train:.4f}, Train Accuracy: {accuracy_train:.2f}%")
            print(f"\t\t Train Accuracy Biased: {accuracy_train_biased:.4f} %, Train Accuracy Unbiased: {accuracy_train_unbiased:.4f} %")
            print(f"\t\t Augmented Samples: {augmented_samples} ({augmented_samples}/{total_train}, {100 * augmented_samples / total_train:.2f} %)")
            print(f"\t -- Validation Loss: {average_loss_val:.4f}, Validation Accuracy: {accuracy_val:.4f} %")
            print(f"\t\t Valid Accuracy Biased: {accuracy_val_biased:.4f} %, Valid Accuracy Unbiased: {accuracy_val_unbiased:.4f} %")
            print(f"\t -- Test Loss: {average_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%")
            print(f"\t\t Test Accuracy Biased: {accuracy_test_biased:.4f}%, Test Accuracy Unbiased: {accuracy_test_unbiased:.4f}%")

        return (
            validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u, 
            test_accuracies_avg, test_accuracies_b, test_accuracies_u
        )


    
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    @staticmethod
    def load_model(filepath, arch_id=50, embedding_dim=None, pretrained=False, target_image_width=224, num_classes=10):
        model = ResNet(arch_id=arch_id, embedding_dim=embedding_dim, pretrained=pretrained, target_image_width=target_image_width, num_classes=num_classes)
        model.load_state_dict(torch.load(filepath))
        return model
    



