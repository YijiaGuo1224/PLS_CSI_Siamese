# Data-Driven Physical Layer Authentication for Wi-Fi
This project develops a deep learning-based physical layer authentication (PLA) scheme for mobile Wi-Fi devices, as detailed in the IEEE TIFS paper entitled **Practical Physical Layer Authentication for Mobile Scenarios Using a Synthetic Dataset Enhanced Deep Learning Approach**. Specifically, a synthetic training dataset is untilized, and a CNN-based Siamese network is adopted for device authentication.

In the training stage, a synthetic training dataset is used to train a Siamese network. In the test stage, the trained Siamese network is applied to experimental test datasets for device authentication.

## Requirements

__I). Development Environment__

This project was developed using the following environment:
- Python 3.8
- TensorFlow 2.10.0 (includes Keras 2.10.0)

__II). Datasets__

The Wi-Fi channel state information (CSI) dataset is available for download [here](https://ieee-dataport.org/documents/wi-fi-channel-state-information-dataset-mobile-physical-layer-authentication). We kindly request that any research or publications resulting from the use of this dataset include a citation to the aforementioned IEEE TIFS paper. Below is the corresponding BibTeX entry:
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

It is worth noting that the experimental training datasets are used to compare against the synthetic training dataset to demonstrate the reliability of using synthetic data.

## Implementation

__I). If you want to train your own Siamese networks__

1) Training Stage

    (i) Extract the downloaded synthetic and experimental training datasets, i.e., 'SyntheticTrainingDataset.zip' and 'ExperimentalTrainingDataset.zip', and place them within the current project file.

    (ii) Creat two files named 'model_synthetic' and 'model_experiment' to store the corresponding trained Siamese networks.

    (iii) Run 'main_syn_train' to train a Siamese network using the synthetic training dataset.

    (iv) Run 'main_exp_train' to train Siamese networks using experimental training datasets collected in 4 typical Wi-Fi indoor scenarios.

2) Test Stage

    (i) Extract the downloaded experimental test datasets, i.e., 'ExperimentalTestDataset.zip', and place it within the current project file.

    (ii) Run 'main_exp_test' to evaluate how the Siamese networks obtained from the above training stage perform on experimental test datasets.

__II). If you simply want to test the Siamese networks instead of training your own ones__

1) Download the trained Siamese network

    (i) The trained Siamese networks are available for download [here](https://ieee-dataport.org/documents/wi-fi-channel-state-information-dataset-mobile-physical-layer-authentication), where the Siamese network trained with the synthetic dataset can be found in “model_synthetic.zip”, and the Siamese networks trained with experimental datasets can be found in “model_experiment.zip”.

    (ii) Extract the downloaded Siamese networks and place them within the current project file.

2) Test Stage

    (i) Extract the downloaded experimental test datasets, i.e., 'ExperimentalTestDataset.zip', and place it within the current project file.

    (i) Run 'main_exp_test' to evaluate the performance of the Siamese networks we provide on experimental test datasets.

## Contact Information
Please contact the following email addresses if you have any questions:

Yijia.Guo@liverpool.ac.uk

Junqing.Zhang@liverpool.ac.uk