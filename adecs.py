
from typing import Any, Iterable, List, Tuple, Union
from ResNet import ResNet
from sklearn.decomposition import PCA
from training_utils import make_prediction, collect_preds_from_features
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import torch

def view_with_pca(tr_features, tr_labels, tr_bias_labels, num_classes, save_results_to):
    pca = PCA(n_components=2).fit(tr_features)
    pca_features = pca.transform(tr_features)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=tr_bias_labels, cmap='jet', alpha=0.5, s=3.0)
    for c in range(num_classes):
        tr_idxs = np.where(tr_labels == c)[0]
        tr_in_class_samples = tr_features[tr_idxs]
        class_centroid = np.mean(tr_in_class_samples, axis=0)
        class_centroid = pca.transform(class_centroid.reshape(1, -1))
        plt.scatter(class_centroid[:, 0], class_centroid[:, 1], marker="X", s=12.0, label=f"Class {c} centroid")
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title('Features Visualization')
    plt.savefig(f"{save_results_to}/PCA_features_visualization.png", dpi=300)
    plt.close()


def configure_adecs(
        model: ResNet,
        train_loader,
        num_classes,
        percentile,
        contamination,    
        gamma,
        bias_amount,
        save_results_to,
        make_figures = True,
        normalization = True
    ) -> Tuple[List[OneClassSVM], List[float], List[float], List[Tuple[float, float]], torch.Tensor, torch.Tensor]:

    bias_detected_ratio = []
    perclass_thresholds = []

    tr_features, tr_labels, tr_bias_labels = model.extract_features(train_loader)

    view_with_pca(tr_features, tr_labels, tr_bias_labels, num_classes, save_results_to)

    adecs = [OneClassSVM(nu=contamination[i], gamma=gamma, kernel="rbf") for i in range(num_classes)]

    tr_cl_predictions, tr_cl_targets, tr_cl_btargets = collect_preds_from_features(
        model, 
        tr_features, 
        tr_labels, 
        tr_bias_labels
    )

    tr_adecs_preds = np.zeros((len(tr_labels), ))
    tr_adecs_scores = np.zeros((len(tr_labels), ))

    for i, adec in enumerate(adecs):
        print(f"Class {i}: ")
        tr_idxs = np.where(tr_labels == i)[0]
        tr_in_class_samples = tr_features[tr_idxs]
        tr_in_class_labels  = tr_labels[tr_idxs]
        tr_in_class_blabels = tr_bias_labels[tr_idxs]

        tr_in_class_preds   = tr_cl_predictions[tr_idxs]
        tr_right_pred = np.where((tr_in_class_preds == tr_in_class_labels))[0]
        tr_fit_features = tr_in_class_samples[tr_right_pred]

        if normalization:
            scaler = MinMaxScaler()
            scaler.fit(tr_fit_features)
            tr_in_class_samples = scaler.transform(tr_in_class_samples)
        
        adec.fit(tr_fit_features)
        df_tr = adec.decision_function(tr_in_class_samples)
        
        perclass_thresholds.append(np.percentile(df_tr, percentile[i]))        
        tr_predictions = np.where(df_tr > perclass_thresholds[i], 1, -1).astype(int) 
        
        tr_adecs_preds[tr_idxs] = tr_predictions
        tr_adecs_scores[tr_idxs] = df_tr
        bias_detected_ratio.append(np.count_nonzero(tr_predictions == -1))
        print(f"Class {i} -- Percentile {percentile[i]} ({np.count_nonzero(tr_predictions == -1)})")

        with open(save_results_to+"/adecs_reports.txt", mode="a+") as f:
            f.write(f"Class {i} -- Percentile {percentile[i]} ({np.count_nonzero(tr_predictions == -1)})\n")

        if make_figures:        
            skip = False
            inc_scores = df_tr[tr_in_class_blabels == 1]
            try:
                ooc_scores = df_tr[tr_in_class_blabels == -1]
            except:
                print("no bias/unbiased distinction, plotting single distribution")
                skip = True
                continue       

            plt.figure()
            plt.hist([inc_scores, ooc_scores], bins=100, label=["Biased", "Unbiased"], alpha=0.7, color=['blue', 'orange'], log=True)
            plt.axvline(perclass_thresholds[i], color='r', linestyle='dashed', linewidth=1, label=f"Percentile {percentile[i]} threshold")
            plt.title(f"Class {i} -Train set- decision function Histogram")
            plt.legend()
            plt.savefig(save_results_to + f"/Hist_Class_{i}_(B={bias_amount})_train_set.png", dpi=300)
            plt.close()

    print("Adecs CF:")
    print(confusion_matrix(tr_bias_labels, tr_adecs_preds))

    return adecs, perclass_thresholds, bias_detected_ratio, None, torch.from_numpy(tr_adecs_preds), torch.from_numpy(tr_adecs_scores)