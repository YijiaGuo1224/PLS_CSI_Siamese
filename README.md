# Data-Driven Physical Layer Authentication for Wi-Fi
This project develops a deep learning-based physical layer authentication (PLA) scheme for mobile devices. Specifically, a synthetic training dataset is untilized, and a CNN-based Siamese network is adopted for device authentication.

In the training stage, a synthetic dataset is used to train a Siamese network. In the test stage, the trained Siamese network is applied to experimental datasets for device authentication.

## Quick start
__I. Requirements__

a) Development Environment

This project was developed using the following environment:
- Python 3.8
- TensorFlow 2.10.0 (includes Keras 2.10.0)

b) Datasets

The Wi-Fi channel state information (CSI) dataset (available for download [here](https://ieee-dataport.org/documents/wi-fi-channel-state-information-dataset-mobile-physical-layer-authentication)) is detailed and utilized in paper entitled **Practical Physical Layer Authentication for Mobile Scenarios Using a Synthetic Dataset Enhanced Deep Learning Approach**, published in IEEE Transactions on Information Forensics and Security.

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

The Wi-Fi CSI dataset contains both experimental and synthetic CSI datasets to evaluate the deep learning-based PLA approaches for mobile Wi-Fi devices. The experimental datasets were collected with two LoPy4 user stations and an ESP32 Wi-Fi receiver in four distinct indoor scenarios, containing detailed packet-level information. The synthetic dataset was generated using MATLAB WLAN Toolbox based on IEEE 802.11 TGn channel model, with controlled channel models, SNR levels, device moving speeds, and distances between transmitters.

__II. Implementation__

a) If you want to train your own Siamese networks

1) Training Stage

    (i) Run 'main_syn_train' to train a Siamese network using the synthetic training dataset.

    (ii) Run 'main_exp_train' to train Siamese networks using experimental training datasets collected in 4 typical Wi-Fi indoor scenarios.

2) Test Stage

    (i) Run 'main_exp_test' to evaluate how the Siamese networks obtained from the above training stage perform on experimental test datasets.

b) If you simply want to test the Siamese networks instead of training your own ones

1) Download the trained Siamese network

    The trained Siamese networks are available for download [here](https://ieee-dataport.org/documents/wi-fi-channel-state-information-dataset-mobile-physical-layer-authentication), where the Siamese network trained with synthetic dataset can be found in the “model_synthetic” directory, and the Siamese networks trained with experimental datasets can be found in the “model_experiment” directory.

2) Test Stage

    (i) Run 'main_exp_test' to evaluate how the Siamese networks we provide perform on experimental test datasets.

## Contact Information
Please contact the following email addresses if you have any questions:

Yijia.Guo@liverpool.ac.uk

Junqing.Zhang@liverpool.ac.uk