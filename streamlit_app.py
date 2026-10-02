import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "plant_disease_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (160, 160)


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="centered"
)


# -----------------------------
# Load Class Names
# -----------------------------
@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)


class_names = load_class_names()
model = load_prediction_model()


# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-5:][::-1]

    results = {
        class_names[i]: float(predictions[i])
        for i in top_indices
    }

    predicted_index = top_indices[0]
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, confidence, results


# -----------------------------
# UI
# -----------------------------
st.title("🌿 Plant Disease Prediction")

st.write(
    "Upload a plant leaf image to identify the possible "
    "disease or healthy category using a deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence, results = predict_disease(image)

        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Top 5 Predictions")

        for disease, probability in results.items():
            st.write(
                f"**{disease}** — {probability * 100:.2f}%"
            )
