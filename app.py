import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image

st.title("PNEUMONIA Detection")

def preprocess_image(image):
    # Resize the image to the required input size of the model
    resized_image = tf.image.resize(image, (256, 256))
    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0
    # Add batch dimension
    processed_image = tf.expand_dims(normalized_image, axis=0)
    return processed_image

def main():
    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Upload an X-Ray chest image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image using st.image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Read and preprocess the uploaded image
        image = Image.open(uploaded_image).convert('RGB')
        processed_image = preprocess_image(image)

        CNN_model_path = os.path.abspath('saved_model_PNEUMONIA_detection.h5')
        model = tf.keras.models.load_model(CNN_model_path)

        # Ensure processed_image has the correct shape (None, 256, 256, 3)
        processed_image = np.array(processed_image)

        yhat = model.predict(processed_image)

        # Display the predictions
        if yhat > 0.5: 
            st.title('PNEUMONIA is Detected!')
        else:
            st.title('Good News! It is Normal.')

if __name__ == "__main__":
    main()
