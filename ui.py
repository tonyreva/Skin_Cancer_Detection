import streamlit as st
import numpy as np
import cv2
import pickle
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


st.set_page_config(
    page_title='cancer detection app',
    page_icon="🤖",
)


with open('image_classification/model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Skin Cancer Detection')
st.write('Upload an image for analysis')
upload = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'webp'])
capture=st.camera_input('capture the image')


if upload is not None:
    # --- 1. Processing for Model Prediction ---
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Create the 28x28 input for the model
    resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    gray_resized = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    normalized = gray_resized.astype('float32') / 255.0
    normalized = normalized.reshape(-1, 28, 28, 1)
    
    # Predict
    output = model.predict(normalized)
    pred = np.argmax(output)
    
    # --- 2. Drawing Logic ---
    # We work on a copy of the original image to draw the rectangle
    display_img = img.copy()
    
    if pred == 1:  # Assuming '1' is the class for Cancer
        st.error('Result: Malignant (Cancerous)')
        
        # Detection Step: Find the dark lesion
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur to ignore hairs/small artifacts
        blurred = cv2.GaussianBlur(gray_full, (7, 7), 0)
        # Threshold: Lesions are usually darker than skin
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest dark area (the lesion)
            largest_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)
            
            # Draw Red Rectangle (BGR format: Red is 0,0,255)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 5)
            
    elif pred == 0:
        st.success('Result: Not having cancer')
    else:
        st.info('Result: Benign (Non-cancerous)')

    # --- 3. Final Display ---
    # Convert BGR (OpenCV) to RGB (Streamlit)
    result_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption='Processed Image', width='stretch')

elif capture is not None:
    raw=np.asarray(bytearray(capture.read()), dtype=np.uint8)
    data=cv2.imdecode(raw, 1)

    st.image(data, 3)
