import streamlit as st
import numpy as np
import os
import cv2
import pickle 
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0'

@st.cache_resource
def load_model():
    with open('image_classification/model.pkl', 'rb') as file:
        return pickle.load(file)

model=load_model()

if 'camera_visible' not in st.session_state:
    st.session_state.camera_visible=False

st.title('SKIN CANCER DETECTION')
st.write('Upload an Image')

st.set_page_config(
    page_title='cancer detection app',
    page_icon="🔬"
)

upload=st.file_uploader('choose the image', type=['jpg', 'jpeg', 'png','webp'])

capture=None
if not st.session_state.camera_visible:
    if st.button('open the camera'):
        st.session_state.camera_visible=True
        st.rerun()
else:
    if st.button('close camera'):
        st.session_state.camera_visible=False
        st.rerun()
    capture=st.camera_input('capture the image')

if upload is not None:
    source=upload
else:
    source=capture

if source is not None:
    raw=np.asarray(bytearray(source.read()), dtype=np.uint8)
    data=cv2.imdecode(raw, 1)
    
    resize=cv2.resize(data, (28, 28), interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    normalized=gray.astype('float32')/255.0
    normalized=normalized.reshape(-1, 28, 28, 1)

    output=model.predict(normalized)
    pred=np.argmax(output)
    
    display_img=data.copy()
    display_img=cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    if pred==1:
        st.error('Malignant (Cancerous)')
        img1=cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(img1, (7,7), 0)
        _, thresh=cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largestCont=max(contours, key=cv2.contourArea)
            x, y, w, h=cv2.boundingRect(largestCont)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (255, 0, 0), 5)

    elif pred==2:
        st.info('Benign (Non-Cancerous)')
        img1=cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(img1, (7,7), 0)

        _, thresh=cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largestCont=max(contours, key=cv2.contourArea)
            x, y, w, h=cv2.boundingRect(largestCont)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 5)

    else:
        st.success('healthy')
    st.image(display_img)
