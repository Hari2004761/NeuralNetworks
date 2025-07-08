# ✋ MNIST Handwritten Digit Recognition

This project trains a neural network to recognize handwritten digits (0–9) from the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using TensorFlow and Keras.

---

## 📚 Dataset Overview

MNIST contains **70,000 grayscale images** of handwritten digits, each sized 28x28 pixels:

- **60,000 images** for training  
- **10,000 images** for testing

Each image is labeled with the digit it represents (0 through 9).

---

## 🧠 Model Description

A simple feedforward neural network was trained that includes:

- Flattening the 2D images into 1D vectors  
- One or more hidden layers with ReLU activation  
- An output layer with softmax activation to classify digits 0–9

Input pixel values were normalized (scaled between 0 and 1) to improve model training.

---

## 🧪 Evaluation

The model achieves high accuracy on the test set (usually above 95%).

### 📊 Confusion Matrix

A confusion matrix was used to analyze model performance:

- Shows how often each digit was correctly predicted  
- Reveals which digits the model confuses most (e.g., 3 vs 5)

---

# 👕 Fashion MNIST Image Classification

This project classifies images of clothing from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using a neural network built with TensorFlow and Keras.

---

## 📚 Dataset Overview

Fashion MNIST consists of **70,000 grayscale images** of size 28x28 pixels, each labeled as one of 10 clothing categories:

| Label | Class        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

- **60,000 images** for training  
- **10,000 images** for testing

---

## 🧠 Model Description

A basic feedforward neural network was trained using:
- Flattened input images
- Two hidden layers with ReLU activation
- An output layer with softmax activation to classify into 10 categories

All pixel values were normalized (scaled between 0 and 1) to improve training performance.

---

## 🧪 Evaluation

The model achieved around **88–89% accuracy** on the test dataset.

### 📊 Confusion Matrix

A confusion matrix was generated to visualize the model’s performance across different clothing categories:

- Diagonal values show correct predictions
- Off-diagonal values show misclassifications

This helps identify which clothing items are often confused (e.g., **shirt vs. T-shirt**, **ankle boot vs. sneaker**).



