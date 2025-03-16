import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn
import tempfile

# Load the trained Xception model
model = timm.create_model('legacy_xception', pretrained=True)

# Modify the classifier to match your task (2 classes: Real/Fake)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load the saved model weights
model.load_state_dict(torch.load('./xception_deepfake_detector_2.pth'))
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Transform for frames
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to predict live video frames and show predictions
def predict_video_live(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Streamlit container for live predictions

    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

            output = model(frame_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()

            label = "REAL" if prediction == 0 else "FAKE"

            # Display frame with prediction
            stframe.image(frame_rgb, caption=f"Prediction: {label} ({confidence:.2f})", use_column_width=True)

    cap.release()

# Streamlit app
st.title("Deepfake Video Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Display uploaded video
    st.video(video_path)

    if st.button("Predict"):
        predict_video_live(video_path)
