# 🌿 Plant Disease Prediction using Deep Learning

An AI-powered plant disease prediction system that uses deep learning and transfer learning to identify plant diseases from leaf images.

The project uses a pretrained EfficientNetB4 model with fine-tuning to classify leaf images into 39 different plant disease and healthy categories.

---

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and agricultural production. This project aims to automate plant disease identification using image classification techniques.

A leaf image is provided as input to the trained deep learning model, which predicts the corresponding disease or healthy category.

The model is trained using a large plant leaf image dataset and achieves **98.36% test accuracy**.

---

## 🚀 Features

- 🌱 Plant disease classification from leaf images
- 🧠 Transfer learning using EfficientNetB4
- 🔍 Classification across 39 different classes
- 🖼️ Image preprocessing and resizing
- 📊 Training and validation performance visualization
- 🔧 Fine-tuning of the pretrained model
- 🎯 Top-5 prediction confidence scores
- 🖥️ Gradio-based prediction interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Split-Folders
- Gradio
- PIL (Python Imaging Library)

---

## 🧠 Model Architecture

The project uses **EfficientNetB4**, pretrained on ImageNet, as the feature extraction backbone.

The architecture consists of:

```text
Input Image
     ↓
160 × 160 × 3
     ↓
EfficientNetB4
     ↓
Global Average Pooling
     ↓
Dropout (0.2)
     ↓
Dense Layer
     ↓
Softmax
     ↓
39 Classes
