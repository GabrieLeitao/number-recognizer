# Handwritten Number Recognizer

This project trains and tests a neural network to recognize handwritten digits (0–9) from **28×28 grayscale images**. It is based on the MNIST dataset but also supports user-drawn digits in a simple OpenCV interface.

---

## Features
- **Model**: Convolutional Neural Network (CNN) built with TensorFlow/Keras.
- **Training**: Uses MNIST dataset with optional data augmentation (rotations, shifts, zoom).
- **Regularization**: Dropout and L2 weight regularization to prevent overfitting.
- **Drawing Interface**: OpenCV canvas to draw digits and get real-time predictions.
- **Preprocessing**: Matches MNIST pipeline by cropping, resizing with preserved aspect ratio, and centering digits.

---

## Preprocessing Pipeline
The MNIST dataset does not store raw drawings. Each digit undergoes:
1. Crop to bounding box.
2. Resize to 20 pixels in the largest dimension, preserving aspect ratio.
3. Embed into a 28×28 canvas, centered by padding.
4. Normalize to `[0,1]` range.


---
