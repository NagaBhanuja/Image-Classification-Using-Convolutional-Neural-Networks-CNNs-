# 🖼️ CIFAR-10 Image Classification with VGG16

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📄 Project Overview
This project implements a **Convolutional Neural Network (CNN)** for image classification tasks using the **CIFAR-10 dataset**. 

To achieve higher accuracy with limited training time, this project utilizes **Transfer Learning**. Specifically, it adapts the **VGG16 architecture** (pre-trained on ImageNet) as a feature extractor, combining it with a custom dense classification head to distinguish between 10 different categories of objects.

## 🎯 Key Features
* [cite_start]**Transfer Learning**: Leverages the powerful feature extraction capabilities of VGG16[cite: 6, 22].
* [cite_start]**Deep Learning Architecture**: Custom fully connected layers added on top of the frozen base model[cite: 33, 34, 35].
* [cite_start]**Optimization**: Uses Stochastic Gradient Descent (SGD) with momentum for stable convergence.
* [cite_start]**Visualization**: Includes prediction visualization with Matplotlib[cite: 130].

## 📊 Dataset Details
[cite_start]The model is trained on the **CIFAR-10** dataset[cite: 5, 12], which includes:
* **Input Shape**: 32x32 pixels (Color/RGB)
* [cite_start]**Classes**: 10 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck) [cite: 127]
* **Data Split**: 50,000 Training images, 10,000 Test images

## 🏗️ Model Architecture
1.  **Input Layer**: 32x32x3 RGB Images.
2.  [cite_start]**Base Model (Frozen)**: VGG16 (Weights: ImageNet)[cite: 22, 26].
3.  [cite_start]**Flatten Layer**: Converts 2D feature maps to vectors[cite: 30].
4.  [cite_start]**Hidden Layer 1**: Dense (256 units, ReLU activation)[cite: 33].
5.  [cite_start]**Hidden Layer 2**: Dense (128 units, ReLU activation)[cite: 34].
6.  [cite_start]**Output Layer**: Dense (10 units, Softmax activation)[cite: 35].

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install tensorflow numpy matplotlib
