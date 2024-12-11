import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page title
st.title('Cat vs Dog Classifier')

# Load the trained model
@st.cache_resource
def load_classifier_model():
    model = load_model('cd_classifier.h5')
    return model

# Function to preprocess the image
def preprocess_image(img):
    # Resize image to 150x150 pixels
    img = img.resize((150, 150))
    # Convert to array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale pixel values
    img_array = img_array / 255.0
    return img_array

# Function to make prediction
def predict(img):
    model = load_classifier_model()
    # Preprocess the image
    processed_img = preprocess_image(img)
    # Make prediction
    prediction = model.predict(processed_img)
    return prediction[0][0]

# Create the Streamlit interface
st.write("Upload an image of a cat or dog, and I'll try to identify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction when user clicks the button
    if st.button('Predict'):
        with st.spinner('Analyzing image...'):
            prediction = predict(img)
            
            # Display results
            st.write('### Prediction Result')
            if prediction > 0.5:
                st.write(f'This is a Dog! (Confidence: {prediction:.2%})')
            else:
                st.write(f'This is a Cat! (Confidence: {(1-prediction):.2%})')

# Add some usage instructions
st.markdown("""
### Instructions:
1. Upload an image of a cat or dog using the file uploader above
2. Click the 'Predict' button to see the classification result
3. The model will tell you whether it's a cat or dog and how confident it is

Note: For best results, use clear images of cats or dogs!
""")