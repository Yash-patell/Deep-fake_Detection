import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn
import tempfile
import time

# Load the trained Xception model
model = timm.create_model('legacy_xception', pretrained=True)

# Modify the classifier to match your task (2 classes: Real/Fake)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load the saved model weights
model.load_state_dict(torch.load('./trained_models/xception_fine_tune_25-frames.pth'))
model.eval()

# Move model to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = torch.load(model_path, map_location=torch.device('cpu'))
model.to(device)

# Transform for frames
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
fixed_fps = 30
frame_time = 1 / fixed_fps
# Function to predict live video frames and show predictions
def predict_video_live(video_path):


    cap = cv2.VideoCapture(video_path)
    # frame_display = st.image([])
    stframe = st.empty()
    sttext = st.empty()

    with torch.no_grad():
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # Model prediction
            output = model(frame_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()

            label = "REAL" if prediction == 0 else "FAKE"
            # stframe.image(frame_rgb, caption=f"Prediction: {label} ({confidence:.2f})",     use_column_width=True)
            # frame_resized = cv2.resize(frame_rgb, (640, 480))
            stframe.image(frame_rgb, use_column_width=True)
            sttext.markdown(f"<h3 style='text-align: center;'>Prediction:  {label} ({confidence:.2f})</h4>", unsafe_allow_html=True)


            # Control frame rate
            elapsed_time = time.time() - start_time
            time.sleep(max(0, frame_time - elapsed_time))

    cap.release()
    # frame_display.empty()
    stframe.empty()
    cv2.destroyAllWindows()
    # frame_display.empty()
# Streamlit app
st.title("Deepfake Video Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Display uploaded video
    # st.video(video_path)

    if st.button("Predict"):
        predict_video_live(video_path)
