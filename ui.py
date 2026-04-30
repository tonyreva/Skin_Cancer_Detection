import streamlit as st
import numpy as np
import cv2
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0'

hide_github_icon = """
    <style>
    .stAppDeployButton {
        display: none !important;
    }
    header {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

@st.cache_resource
def load_model():
   with open('image_classification/new_model.pkl', 'rb') as file:
      return pickle.load(file)
      
model=load_model()

if 'camera_visible' not in st.session_state:
   st.session_state.camera_visible=False
 
if 'run_live' not in st.session_state:
   st.session_state.run_live=False

st.title('SKIN CANCER DETECTION')
st.write('upload image to detect skin cancer')

st.set_page_config(
page_title='cancer detection app',
page_icon="🔬"
)


class LesionProcessor(VideoProcessorBase):
   def __init__(self):
      self.frame_count = 0
      self.skip_frames = 1
      self.is_cancer = False
      self.last_box = None
      self.face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
      self.body_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_fullbody.xml')

   def recv(self, frame):
      img = frame.to_ndarray(format="bgr24")
      gray_small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces=self.face_cascade.detectMultiScale(gray_small, 1.3, 5)
      bodies=self.body_cascade.detectMultiScale(gray_small, 1.1, 3)
      self.frame_count += 1
        
      if len(faces)>0 or len(bodies)>0:
         self.is_cancer=False
         self.last_box=None
         return av.VideoFrame.from_ndarray(img, format='bgr24')

      if self.frame_count % self.skip_frames == 0:
          resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
          gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
          normalized = gray.astype('float32') / 255.0
          normalized = normalized.reshape(1, 28, 28, 1)
            
            
          output = model.predict(normalized, verbose=0)
          confidence=np.max(output)
          pred = np.argmax(output)
          self.is_cancer = (pred == 1 and confidence >0.50)
            
          if self.is_cancer:
             gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             blurred = cv2.GaussianBlur(gray_full, (7, 7), 0)
             _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
             contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
             if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 500:
                   self.last_box = cv2.boundingRect(cnt)
                else:
                   self.last_box = None
             else:
                self.last_box = None

       
      display_img = img.copy()
      if self.is_cancer and self.last_box:
         x, y, w, h = self.last_box
         cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

      return av.VideoFrame.from_ndarray(display_img, format="bgr24")




upload=st.file_uploader('choose the image')
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
   confidence=np.max(output)
   pred=np.argmax(output)
   
   display_img=data.copy()
   display_img=cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
   if pred==1:
      st.error('Malignant (Cancerous)')
      img1=cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      blurred=cv2.GaussianBlur(img1, (7, 7), 0)
      _, thresh=cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      contours, _=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      if contours:
         largestCont=max(contours, key=cv2.contourArea)
         x, y, w, h=cv2.boundingRect(largestCont)
         cv2.rectangle(display_img, (x, y), (x+w, y+h), (255, 0, 0), 5)
 
 
   elif pred==2:
      st.info('Benign (Non-Cancerous)')
      img=cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      blurred=cv2.GaussianBlur(img, (7,7), 0)
      
      _, thresh=cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      contours, _=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
      if contours:
         largestCont=max(contours, key=cv2.contourArea)
         if cv2.contourArea(largestCont)>500:
            x, y, w, h=cv2.boundingRect(largestCont)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 5)
          

   else:
      st.success('healthy')
  
   st.image(display_img)
  

else:

   if not st.session_state.run_live:
      if st.button('live detection'):
         st.session_state.run_live=True
         st.rerun()
   else:
      if st.button('stop live'):
         st.session_state.run_live=False
         st.rerun()
      
   
      webrtc_streamer(
      key="lesion-live",
      video_processor_factory=LesionProcessor,
      media_stream_constraints={'video': True, 'audio': False},
      rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
      },
      async_processing=True
     )
     

