import streamlit as st
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

# Load your pre-trained model
model = load_model('final_model.h5')  # Replace 'my_model.h5' with your actual model file name

# Define the size you used during training
SIZE = 256  # Ensure this matches the size used for training

# Streamlit UI
st.title("Image Super-Resolution App")

# File uploader for low-resolution image
uploaded_file = st.file_uploader("Upload a low-resolution image", type=["jpg", "png", "jpeg"])

# Function to plot images in Streamlit
def plot_images(high, low, predicted):
    st.image(high, caption='High Resolution Image', use_column_width=True)
    st.image(low, caption='Low Resolution Image', use_column_width=True)
    st.image(predicted, caption='Predicted Image', use_column_width=True)

if uploaded_file is not None:
    # Open and process the uploaded image
    input_image = Image.open(uploaded_file)
    
    # Resize the image to the required size
    input_image = input_image.resize((SIZE, SIZE))
    
    # Convert to array and normalize
    input_array = img_to_array(input_image) / 255.0
    input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension

    # Make prediction
    predicted_image = model.predict(input_array)[0]
    
    # Clip predicted image to [0, 1] range
    predicted_image = np.clip(predicted_image, 0.0, 1.0)

    # Convert back to the original image format for display
    predicted_image = (predicted_image * 255).astype(np.uint8)  # Scale back to [0, 255]
    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format if needed
    
    # Display the images
    plot_images(input_image, input_image, predicted_image)


