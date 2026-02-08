import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import HybridDetector
from pathlib import Path

# Configuration
DATA_ROOT = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\processed_data"
CHECKPOINT_DIR = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 4 # Small batch size for laptop GPU/CPU
EPOCHS = 10
LEARNING_RATE = 1e-4
FRAMES_PER_VIDEO = 10

class DeepFakeDataset(Dataset):
    def __init__(self, split='train', root_dir=DATA_ROOT):
        self.root_dir = root_dir
        self.split = split
        self.samples = []
        
        # Load data
        # Structure: root/split/label/faces/vid_frame_X.jpg
        # We need to group by video.
        
        for label in ['real', 'fake']:
            face_dir = os.path.join(root_dir, split, label, 'faces')
            audio_dir = os.path.join(root_dir, split, label, 'audio')
            
            # Simple grouping by filename prefix
            # Files: vidname_frame_0.jpg
            # Audio: vidname_audio.npy
            
            # Get unique video stems
            audio_files = list(Path(audio_dir).glob("*.npy"))
            
            for aud_path in audio_files:
                stem = aud_path.stem.replace('_audio', '')
                
                # Check if we have frames
                frame_pattern = os.path.join(face_dir, f"{stem}_frame_*.jpg")
                frames = sorted(glob.glob(frame_pattern))
                
                if len(frames) > 0 and os.path.exists(aud_path):
                    self.samples.append({
                        'stem': stem,
                        'frames': frames,
                        'audio': str(aud_path),
                        'label': 0 if label == 'real' else 1
                    })
        
        print(f"Loaded {len(self.samples)} samples for {split} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process Frames
        frame_tensors = []
        # Ensure we have fixed number of frames. 
        # If less, loop. If more, sample.
        current_frames = sample['frames']
        
        if len(current_frames) > FRAMES_PER_VIDEO:
            current_frames = current_frames[:FRAMES_PER_VIDEO]
            
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for fpath in current_frames:
            img = cv2.imread(fpath)
            if img is None:
                # Placeholder black frame
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_tensors.append(transform(img))
            
        # Pad if missing frames
        while len(frame_tensors) < FRAMES_PER_VIDEO:
            frame_tensors.append(frame_tensors[-1] if len(frame_tensors) > 0 else torch.zeros(3, 224, 224))
            
        video_tensor = torch.stack(frame_tensors) # (T, 3, 224, 224)
        
        # Process Audio
        spec = np.load(sample['audio']) # (128, Time)
        # Resize/Crop to fixed time dimension
        target_time = 313 # approx 5s at hop 512
        if spec.shape[1] > target_time:
            spec = spec[:, :target_time]
        else:
            pad_width = target_time - spec.shape[1]
            spec = np.pad(spec, ((0,0), (0, pad_width)), mode='constant')
            
        audio_tensor = torch.FloatTensor(spec).unsqueeze(0) # (1, 128, 313)
        
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        return video_tensor, audio_tensor, label_tensor

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = DeepFakeDataset('train')
    val_dataset = DeepFakeDataset('val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = HybridDetector(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for vid, aud, labels in tqdm(train_loader):
            vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(vid, aud)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vid, aud, labels in val_loader:
                vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
                outputs = model(vid, aud)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            
if __name__ == "__main__":
    train_model()
