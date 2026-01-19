# Image Classification Using Convolutional Neural Networks (CNNs)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white)

## 📌 Project Overview
This project implements an image classification system using **Convolutional Neural Networks (CNNs)**. It utilizes **Transfer Learning** with the pre-trained **VGG16** architecture to classify images from the **CIFAR-10** dataset into 10 distinct categories.

The primary goal is to demonstrate how to leverage a powerful pre-trained feature extractor (VGG16) and adapt it for a new classification task by adding custom fully connected layers.

## 📂 Dataset
The project utilizes the **CIFAR-10** dataset, a standard benchmark in computer vision.
* **Input Dimensions:** 32x32 pixels (Color/RGB)
* **Training Set:** 50,000 images
* **Test Set:** 10,000 images
* **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## 🧠 Model Architecture
The model uses **VGG16** (trained on ImageNet) as the base, with its weights frozen to preserve learned features. A custom classification head is added on top:

1.  **Base Model:** VGG16 (exclude top layers, input shape 32x32x3)
2.  **Flatten Layer:** Converts 2D feature maps into a 1D vector
3.  **Dense Layer:** 256 neurons (ReLU activation)
4.  **Dense Layer:** 128 neurons (ReLU activation)
5.  **Output Layer:** 10 neurons (Softmax activation)

## ⚙️ Training Configuration
* **Optimizer:** SGD (Stochastic Gradient Descent)
    * Learning Rate: 0.001
    * Momentum: 0.9
* **Loss Function:** Categorical Crossentropy
* **Epochs:** 10
* **Batch Size:** 32

## 📊 Results
Upon evaluation on the test set, the model achieves an accuracy of approximately **60%** after 10 epochs.

* **Test Loss:** ~1.13
* **Test Accuracy:** ~60.5%

## 🚀 How to Run
1.  **Prerequisites:** Ensure Python, TensorFlow, and Matplotlib are installed.
    ```bash
    pip install tensorflow numpy matplotlib
    ```
2.  **Execution:** Run the provided notebook or Python script.
    ```bash
    python image_classification_using_cnns.py
    ```
3.  **Prediction:** The script includes a demonstration where it predicts the class of a single test image (e.g., classifying a "Ship") and visualizes it.

---
*This project is for educational purposes to demonstrate Transfer Learning with Keras.*
