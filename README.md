# Physical Layer Security using a Siamese Network and Synthetic Dataset
This project develops a deep learning-based physical layer authentication (PLA) scheme for mobile devices. Specifically, a synthetic training dataset is untilized, and a CNN-based Siamese network is adopted for device authentication.

In the training stage, a synthetic dataset is used to train a Siamese network. In the test stage, the trained Siamese network is applied to experimental datasets for device authentication.

## Quick start
__1. Requirements__

a) Development Environment

This project was developed using the following environment:
- Python 3.8
- TensorFlow 2.10.0 (includes Keras 2.10.0)

b) Download Dataset

The Wi-Fi channel state information (CSI) dataset (available for download [here](https://ieee-dataport.org/documents/wi-fi-channel-state-information-dataset-mobile-physical-layer-authentication) is detailed and utilized in paper entitled **Practical Physical Layer Authentication for Mobile Scenarios Using a Synthetic Dataset Enhanced Deep Learning Approach**, published in IEEE Transactions on Information Forensics and Security.

The dataset is available for research purposes. We kindly request that any research or publications resulting from the use of this dataset include a citation to the aforementioned IEEE TIFS paper. Below is the corresponding BibTeX entry:
```bibtex
@ARTICLE{guo2025practical,
author={Guo, Yijia and Zhang, Junqing and Hong, Y.-W. Peter},
journal={IEEE Trans. Inf. Forensics Secur.}, 
title={Practical physical layer authentication for mobile scenarios using a synthetic dataset enhanced deep learning approach}, 
year={2025},
volume={20},
number={},
pages={9305-9317}
}
```

c) Training Stage

Run 'main_syn_train' to train a Siamese network using the synthetic trainingn dataset.

Run 'main_exp_train' to train a Siamese network using experimental training datasets collected in 4 typical Wi-Fi indoor scenarios.

d) Test Stage

Run 'main_exp_test' to evaluate the performance of the Siamese network trained on the synthetic dataset using the experimental test datasets.

## Contact Information
Please contact the following email addresses if you have any questions:

Yijia.Guo@liverpool.ac.uk

Junqing.Zhang@liverpool.ac.uk