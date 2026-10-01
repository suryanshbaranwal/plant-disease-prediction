import json
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "plant_disease_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (160, 160)

# -----------------------------
# Load class names
# -----------------------------
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# -----------------------------
# Load trained model
# -----------------------------
model = load_model(MODEL_PATH)


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image):
    if image is None:
        return None

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# -----------------------------
# Prediction function
# -----------------------------
def predict_disease(image):
    if image is None:
        return "Please upload a leaf image.", {}

    img = preprocess_image(image)

    predictions = model.predict(img, verbose=0)[0]

    top_indices = np.argsort(predictions)[-5:][::-1]

    results = {
        class_names[i]: float(predictions[i])
        for i in top_indices
    }

    predicted_index = top_indices[0]
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    result = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"

    return result, results


# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="Plant Disease Prediction") as demo:

    gr.Markdown(
        """
        # 🌿 Plant Disease Prediction

        Upload a plant leaf image to identify the possible
        disease or healthy category using a deep learning model.
        """
    )

    image_input = gr.Image(
        type="pil",
        label="Upload Leaf Image"
    )

    predict_button = gr.Button(
        "🔍 Predict Disease"
    )

    prediction_output = gr.Textbox(
        label="Prediction"
    )

    confidence_output = gr.Label(
        label="Top Predictions",
        num_top_classes=5
    )

    predict_button.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=[prediction_output, confidence_output]
    )


# -----------------------------
# Launch application
# -----------------------------
if __name__ == "__main__":
    demo.launch()
