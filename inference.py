import torch
import cv2
import numpy as np
import librosa
import argparse
import os
from torchvision import transforms
from model import HybridDetector

# Configuration
CHECKPOINT_PATH = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\checkpoints\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_VIDEO = 10
AUDIO_DURATION = 5
SAMPLE_RATE = 16000

# Global model cache
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
        
    model = HybridDetector(num_classes=2).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {CHECKPOINT_PATH}")
        _model = model
        return model
    else:
        print("Error: Checkpoint not found.")
        return None

def preprocess_video_for_ui(video_path):
    """
    Returns processed tensors AND raw artifacts for UI visualization.
    """
    # 1. Extract Faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, FRAMES_PER_VIDEO, dtype=int)
    
    frame_tensors = []
    face_crops_rgb = [] # For UI display
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            # Crop logic
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2*margin)
            h = min(frame.shape[0] - y, h + 2*margin)
            face_img = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
        else:
             face_img = cv2.resize(frame, (224, 224))

        face_crops_rgb.append(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        frame_tensors.append(transform(face_img_rgb))
        
    cap.release()
    
    # Pad
    while len(frame_tensors) < FRAMES_PER_VIDEO:
         frame_tensors.append(frame_tensors[-1] if len(frame_tensors) > 0 else torch.zeros(3, 224, 224))
         
    video_tensor = torch.stack(frame_tensors).unsqueeze(0)
    
    # 2. Extract Audio
    audio_spec_db = None
    try:
        y, sr = librosa.load(video_path, sr=SAMPLE_RATE, duration=AUDIO_DURATION)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        audio_spec_db = S_dB # For UI
        
        target_time = 313
        if S_dB.shape[1] > target_time:
            S_dB = S_dB[:, :target_time]
        else:
            pad_width = target_time - S_dB.shape[1]
            S_dB = np.pad(S_dB, ((0,0), (0, pad_width)), mode='constant')
            
        audio_tensor = torch.FloatTensor(S_dB).unsqueeze(0).unsqueeze(0)
        
    except Exception:
        audio_tensor = torch.zeros((1, 1, 128, 313))
        audio_spec_db = np.zeros((128, 313))
        
    return video_tensor, audio_tensor, face_crops_rgb, audio_spec_db

def analyze_video(video_path):
    model = load_model()
    if model is None: return None
    
    vid_t, aud_t, faces, spec = preprocess_video_for_ui(video_path)
    if vid_t is None: return None
    
    vid_t = vid_t.to(DEVICE)
    aud_t = aud_t.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(vid_t, aud_t)
        probs = torch.softmax(outputs, dim=1)
        fake_prob = probs[0][1].item()
        
    return {
        'fake_probability': fake_prob,
        'label': "FAKE" if fake_prob > 0.5 else "REAL",
        'faces': faces,
        'spectrogram': spec
    }

def get_frame_importance(video_tensor, audio_tensor):
    """
    Computes magnitude of gradients per frame to estimate "Importance".
    Returns a list of scalar scores (one per frame).
    """
    model = load_model()
    if model is None: return None

    # Enable gradients
    video_tensor.requires_grad_()
    audio_tensor.requires_grad_()
    
    # Forward pass
    outputs = model(video_tensor, audio_tensor)
    
    # Target class (Max logit)
    score_max_index = outputs.argmax()
    score = outputs[0, score_max_index]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Get gradients: (1, T, 3, 224, 224)
    gradients = video_tensor.grad 
    if gradients is None: return None
    
    gradients = gradients.squeeze(0) # (T, 3, 224, 224)
    
    # Compute importance per frame: Sum of absolute gradients
    # (T, 3, 224, 224) -> (T,)
    importance_scores = torch.sum(gradients.abs(), dim=[1, 2, 3])
    
    # Normalize to 0-1 for plotting
    importance_scores = importance_scores.cpu().detach().numpy()
    if importance_scores.max() > 0:
        importance_scores = importance_scores / importance_scores.max()
        
    return importance_scores.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    args = parser.parse_args()
    
    result = analyze_video(args.video_path)
    if result:
        print(f"Fake Prob: {result['fake_probability']:.4f}")
        print(f"Prediction: {result['label']}")
