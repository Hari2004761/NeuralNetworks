# 🧠 Handwritten Digit Recognition with TensorFlow & Keras

This project uses a neural network to recognize handwritten digits (0–9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using TensorFlow and Keras.

---

## 📂 Dataset

- **MNIST** contains 70,000 grayscale images of handwritten digits.
- 60,000 images for training, 10,000 for testing.
- Each image is 28x28 pixels.

---

## 🚀 Model Overview

The model is a simple feedforward neural network (Multi-Layer Perceptron):

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
