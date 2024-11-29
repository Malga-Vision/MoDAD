# Looking at Model Debiasing through the Lens of Anomaly Detection
Official Pytorch source code supporting the implementation of:  
<i>"Looking at Model Debiasing through the Lens of Anomaly Detection",  
<u>Vito Paolo Pastore, Massimiliano Ciranni</u>, Davide Marinelli, Francesca Odone, Vittorio Murino,
IEEE/CVF WACV, 2025</i>

[ArXiv Preprint](https://arxiv.org/abs/2407.17449)

<br>

## Abstract
<i>
Deep neural networks are likely to learn unintended spurious correlations between training data and labels when dealing with biased data, potentially limiting the generalization to unseen samples not presenting the same bias. In this context, model debiasing approaches can be devised aiming at reducing the model’s dependency on such unwanted correlations, either leveraging the knowledge of bias information or not. In this work, we focus on the latter and more realistic scenario, showing the importance of accurately predicting the bias-conflicting and bias-aligned samples to obtain compelling performance in bias mitigation. On this ground, we propose to conceive the problem of model bias from an out-of-distribution perspective, introducing a new bias identification method based on anomaly detection. We claim that when data is mostly biased, bias-conflicting samples can be regarded as outliers with respect to the bias-aligned distribution in the feature space of a biased model, thus allowing for precisely detecting them with an anomaly detection method. Coupling the proposed bias identification approach with bias-conflicting data upsampling and augmentation in a two-step strategy, we reach state-of-the-art performance on synthetic and real benchmark datasets. Ultimately, our proposed approach shows
that the data bias issue does not necessarily require complex debiasing methods, given that an accurate bias identification procedure is defined. 
</i>

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/74a87f3f-b9e0-4913-a372-603c8fd3acce" style="width:640px; height:360px;">
</p>

<br>

### Source Code

This source code contains all the datasets, model implementations, implemented methods, and experiment routines.
All the available operations are exposed through the file `experiment_utils.py`.

Datasets are expected to be stored in the `data` folder.

### Citing our work
If you happen to find our work useful for your research, please cite us as:
```bibtex
@misc{pastore2024lookingmodeldebiasinglens,
      title={Looking at Model Debiasing through the Lens of Anomaly Detection}, 
      author={Vito Paolo Pastore and Massimiliano Ciranni and Davide Marinelli and Francesca Odone and Vittorio Murino},
      year={2024},
      eprint={2407.17449},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.17449}, 
}
```
