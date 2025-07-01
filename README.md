# Mobile-PLA
This project develops a deep learning-based physical layer authentication (PLA) scheme for mobile devices. Specifically, a synthetic training dataset is untilized, and a CNN-based Siamese network is adopted for device authentication.

In the training stage, a synthetic dataset is used to train a Siamese network. In the test stage, the trained Siamese network is applied to experimental datasets for device authentication.

## Quick start
__1. Requirements__

a) Development Environment

This project was developed using the following environment:
- Python 3.8
- TensorFlow 2.10.0 (includes Keras 2.10.0)

b) Download Dataset

c) Training Stage

Run 'main_syn_train' to train a Siamese network using the synthetic trainingn dataset.

Run 'main_exp_train' to train a Siamese network using experimental training datasets collected in 4 typical Wi-Fi indoor scenarios.

d) Test Stage

Run 'main_exp_test' to evaluate the performance of the Siamese network trained on the synthetic dataset using the experimental test datasets.

## Contact Information
Please contact the following email addresses if you have any questions:

Yijia.Guo@liverpool.ac.uk

Junqing.Zhang@liverpool.ac.uk