# Alzheimer's Disease Classification Using ResNet50

A CNN-based deep learning project for classifying brain MRI images into four categories using ResNet50 Transfer Learning.

## Classes

The model classifies MRI images into:

- No Impairment
- Very Mild Impairment
- Mild Impairment
- Moderate Impairment

## Model

The project uses:

- ResNet50
- ImageNet pretrained weights
- Transfer Learning
- Data Augmentation
- Batch Normalization
- Dropout
- Softmax Classification

### Architecture

```text
MRI Image
    |
    v
Data Augmentation
    |
    v
ResNet50
    |
    v
Global Average Pooling
    |
    v
Batch Normalization
    |
    v
Dropout
    |
    v
Dense Layer (256)
    |
    v
Batch Normalization
    |
    v
Dropout
    |
    v
Softmax
    |
    v
4 Classes
