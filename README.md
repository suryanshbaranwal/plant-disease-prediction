# 🌿 Plant Disease Prediction using Deep Learning

An AI-powered plant disease prediction system that uses deep learning and transfer learning to identify plant diseases from leaf images.

The project uses a pretrained **EfficientNetB4** model with fine-tuning to classify plant leaf images into **39 disease and healthy categories**.

## 🚀 Live Demo

👉 https://suryansh-plant-disease-prediction.streamlit.app/

## 🖥️ Live App Preview

![Plant Disease Prediction App](./plant-disease-demo.png.png)

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and agricultural production. This project aims to automate plant disease identification using image classification techniques.

Users can upload a plant leaf image, and the trained deep learning model predicts the corresponding disease or healthy category along with confidence scores.

The model is trained on a large plant leaf image dataset and uses transfer learning with EfficientNetB4 for image classification.

## ✨ Features

- 🌱 Plant disease classification from leaf images
- 🧠 Transfer learning using EfficientNetB4
- 🏷️ Classification across 39 categories
- 🖼️ Image preprocessing and resizing
- 📊 Top-5 prediction confidence scores
- 🔍 Fine-tuned deep learning model
- 🌐 Interactive Streamlit web application
- ⚡ Real-time prediction from uploaded images

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Split-Folders
- Streamlit
- Git & GitHub
- Git LFS

## 🧠 Model Architecture

The project uses **EfficientNetB4** as the feature extraction backbone with a custom classification layer.

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

📊 Dataset

The dataset contains plant leaf images belonging to 39 different categories, including healthy and diseased plants.

The categories cover multiple crops such as:

🍎 Apple
🫐 Blueberry
🍒 Cherry
🌽 Corn
🍇 Grape
🍊 Orange
🍑 Peach
🌶️ Pepper
🥔 Potato
Raspberry
Squash
🍓 Strawberry
🍅 Tomato
🔬 Prediction Workflow
Upload Leaf Image
        ↓
Image Preprocessing
        ↓
Resize to 160 × 160
        ↓
Deep Learning Model
        ↓
Disease Classification
        ↓
Prediction + Confidence Score
📈 Model Performance

The trained model achieved approximately 98.36% test accuracy during evaluation.

The application also displays the Top-5 predictions with confidence scores to provide additional prediction information.

💻 Run Locally
1. Clone the repository
git clone https://github.com/suryanshbaranwal/plant-disease-prediction.git
cd plant-disease-prediction
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit application
streamlit run streamlit_app.py

The application will open in your browser.

📁 Project Structure
plant-disease-prediction/
│
├── app.py
├── streamlit_app.py
├── class_names.json
├── plant_disease_cnn_model.keras
├── plant_disease_prediction.ipynb
├── requirements.txt
├── README.md
├── .gitattributes
└── plant-disease-demo.png.png
🌐 Deployment

The application is deployed using Streamlit Community Cloud.

The trained model is stored using Git LFS because of its large file size.

Live Application

👉 https://suryansh-plant-disease-prediction.streamlit.app/

🔮 Future Improvements
📱 Develop a mobile-friendly version
🌿 Add more plant disease categories
💊 Provide disease treatment and prevention suggestions
📷 Support real-time camera-based leaf detection
📊 Add detailed disease information
🤖 Improve model accuracy with additional datasets
👨‍💻 Author
Suryansh Baranwal

Computer Science & Engineering student specializing in Artificial Intelligence and Machine Learning.

🔗 GitHub:
https://github.com/suryanshbaranwal

🔗 Project Repository:
https://github.com/suryanshbaranwal/plant-disease-prediction
