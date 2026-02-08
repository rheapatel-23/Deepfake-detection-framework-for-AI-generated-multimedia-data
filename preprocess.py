import os
import cv2
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import subprocess

# Configuration
DATASET_ROOT = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\dataset"
OUTPUT_ROOT = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\processed_data"
FRAMES_PER_VIDEO = 10  # Number of face frames to extract per video
AUDIO_DURATION = 5  # Seconds of audio to analyze
SAMPLE_RATE = 16000

def extract_faces(video_path, output_dir):
    """
    Extracts face frames from a video.
    Uses OpenCV's DNN face detector or Haar Cascades.
    For simplicity and speed without extra weights, we'll use Haar first.
    """
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return
        
    # Sample frames uniformly
    indices = np.linspace(0, total_frames-1, FRAMES_PER_VIDEO, dtype=int)
    
    saved_count = 0
    fn = video_path.stem
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # If faces found, save the largest one
        if len(faces) > 0:
            # Sort by area
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Expand margin slightly
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2*margin)
            h = min(frame.shape[0] - y, h + 2*margin)
            
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (224, 224)) # Resize for CNN
                save_path = os.path.join(output_dir, f"{fn}_frame_{saved_count}.jpg")
                cv2.imwrite(save_path, face_crop)
                saved_count += 1
                
    cap.release()

def extract_audio_spectrogram(video_path, output_dir):
    """
    Extracts audio from video and generates a Mel-Spectrogram.
    """
    try:
        # 1. Extract audio to temp file using ffmpeg (via moviepy or subprocess)
        # We'll use librosa directly if it can handle the container, 
        # but often it's safer to separate audio first. 
        # For this environment, let's try reading directly with librosa (uses ffmpeg backend)
        
        # Determine path
        audio_path = str(video_path)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=AUDIO_DURATION)
        
        if len(y) == 0:
            return

        # Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Save as image (for CNN) or numpy array
        # Saving as numpy array is better for ML, but image is easier to visualize/debug initially.
        # Let's save as numpy for the dataloader.
        
        save_path = os.path.join(output_dir, f"{video_path.stem}_audio.npy")
        np.save(save_path, S_dB)
        
        # Optional: Save image for visualization
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(S_dB, sr=sr)
        # plt.savefig(os.path.join(output_dir, f"{video_path.stem}_spec.png"))
        # plt.close()
        
    except Exception as e:
        # print(f"Audio error for {video_path}: {e}")
        pass

def process_dataset():
    print("Starting Feature Extraction...")
    
    splits = ['train', 'val', 'test']
    labels = ['real', 'fake']
    
    for split in splits:
        for label in labels:
            input_dir = os.path.join(DATASET_ROOT, split, label)
            
            # Create output dirs
            face_out = os.path.join(OUTPUT_ROOT, split, label, 'faces')
            audio_out = os.path.join(OUTPUT_ROOT, split, label, 'audio')
            
            os.makedirs(face_out, exist_ok=True)
            os.makedirs(audio_out, exist_ok=True)
            
            videos = list(Path(input_dir).glob("*.mp4"))
            
            print(f"Processing {split}/{label} - {len(videos)} videos")
            
            for vid in tqdm(videos):
                extract_faces(vid, face_out)
                extract_audio_spectrogram(vid, audio_out)

if __name__ == "__main__":
    process_dataset()
