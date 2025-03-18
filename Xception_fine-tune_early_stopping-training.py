# Xception fine tune with early stopping, patience = 3, fixed frames = 24, bs = 4


import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import ssl
from tqdm import tqdm
import timm
import torch.cuda.amp as amp  # For mixed precision

# Paths to real and fake video folders
real_videos_path = './Celeb-real'
fake_videos_path = './Celeb-fake'

# Define frame extraction function
def extract_frames(video_path, num_frames=24):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        num_frames = total_frames

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

    cap.release()
    return frames

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, real_videos_path, fake_videos_path, transform=None):
        self.real_videos = [(os.path.join(real_videos_path, vid), 0) for vid in os.listdir(real_videos_path)]
        self.fake_videos = [(os.path.join(fake_videos_path, vid), 1) for vid in os.listdir(fake_videos_path)]
        self.data = self.real_videos + self.fake_videos
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = extract_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        return frames, label

# Transform for frames
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset and DataLoader
dataset = DeepfakeDataset(real_videos_path, fake_videos_path, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# Load pretrained Xception model from timm
model = timm.create_model('xception', pretrained=True, num_classes=2)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Early stopping setup
patience = 3
best_val_loss = float('inf')
stopping_step = 0

num_epochs = 25
scaler = amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for frames, labels in progress_bar:
        frames = frames.to(device)
        labels = labels.to(device)

        batch_size, num_frames, c, h, w = frames.shape
        outputs = None

        optimizer.zero_grad()

        for i in range(num_frames):
            frame = frames[:, i].to(device)

            with amp.autocast():
                frame_outputs = model(frame)

            if outputs is None:
                outputs = frame_outputs
            else:
                outputs += frame_outputs

            torch.cuda.empty_cache()

        outputs /= num_frames

        loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

    # Early stopping check
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        stopping_step = 0
        torch.save(model.state_dict(), 'xception_fine_tune_25-frames.pth')
    else:
        stopping_step += 1

    if stopping_step >= patience:
        print("Early stopping triggered")
        break
