import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('model/brain_tumor_cnn.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.set_page_config(page_title="Brain Tumor Detection", layout='centered')

# Title
st.title("Brain Tumor Detection using CNN")

# File uploader
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    # Show result
    st.success(f' Predicted Tumor Type: **{predicted_label.upper()}**')
