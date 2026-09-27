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

## Model Architecture

The proposed model uses a ResNet50-based Convolutional Neural Network with transfer learning.

```text
Input MRI Image (224 × 224 × 3)
                |
                v
        Data Augmentation
     Flip | Rotation | Zoom
                |
                v
          Normalization
             1 / 255
                |
                v
        +----------------+
        |    ResNet50    |
        | ImageNet       |
        | Pretrained     |
        +----------------+
                |
        Fine-Tune Last 10
             Layers
                |
                v
   Global Average Pooling
                |
                v
     Batch Normalization
                |
                v
         Dropout (0.4)
                |
                v
       Dense Layer (256)
        ReLU Activation
                |
                v
     Batch Normalization
                |
                v
         Dropout (0.4)
                |
                v
       Dense Layer (4)
       Softmax Activation
                |
                v
        Classification
                |
       +--------+--------+
       |        |        |
       v        v        v
      No      Very      Mild/
   Impairment  Mild    Moderate
Softmax
    |
    v
4 Classes

##Dataset Structure

alzheimer_dataset/
│
├── train/
│   ├── No Impairment/
│   ├── Very Mild Impairment/
│   ├── Mild Impairment/
│   └── Moderate Impairment/
│
└── test/
    ├── No Impairment/
    ├── Very Mild Impairment/
    ├── Mild Impairment/
    └── Moderate Impairment/
