import os
import shutil
import random
import glob
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_ROOT = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\FakeAVCeleb_v1.2"
TARGET_ROOT = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\dataset"

# Recommendation: Start with ~600 videos total for quick iteration.
# Increase this number (e.g., to 2000 or 5000) once the pipeline is verified.
TOTAL_SAMPLES = 600 
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def get_video_files(directory):
    """Recursively find all .mp4 files in a directory."""
    return list(Path(directory).rglob("*.mp4"))

def setup_directories():
    """Create the target directory structure."""
    if os.path.exists(TARGET_ROOT):
        print(f"Warning: Target directory {TARGET_ROOT} already exists.")
        # Optional: shutil.rmtree(TARGET_ROOT) to clear it
    
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(TARGET_ROOT, split, label), exist_ok=True)

def main():
    print("Scanning source directory...")
    
    # Define categories based on directory names
    # Real = RealVideo-RealAudio
    # Fake = FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio
    
    real_dir = os.path.join(SOURCE_ROOT, "RealVideo-RealAudio")
    fake_dirs = [
        os.path.join(SOURCE_ROOT, "FakeVideo-FakeAudio"),
        os.path.join(SOURCE_ROOT, "FakeVideo-RealAudio"),
        os.path.join(SOURCE_ROOT, "RealVideo-FakeAudio")
    ]
    
    real_videos = get_video_files(real_dir)
    fake_videos = []
    for d in fake_dirs:
        fake_videos.extend(get_video_files(d))
        
    print(f"Found {len(real_videos)} Real videos.")
    print(f"Found {len(fake_videos)} Fake videos.")
    
    # Sampling
    samples_per_class = TOTAL_SAMPLES // 2
    
    if len(real_videos) < samples_per_class or len(fake_videos) < samples_per_class:
        print(f"Error: Not enough videos to satisfy {samples_per_class} per class.")
        return

    sampled_real = random.sample(real_videos, samples_per_class)
    sampled_fake = random.sample(fake_videos, samples_per_class)
    
    # Shuffle
    random.shuffle(sampled_real)
    random.shuffle(sampled_fake)
    
    # Calculate split indices
    n_train = int(samples_per_class * SPLIT_RATIOS['train'])
    n_val = int(samples_per_class * SPLIT_RATIOS['val'])
    # Remaining go to test to ensure total matches
    
    splits = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, samples_per_class)
    }
    
    setup_directories()
    
    print(f"Copying files to {TARGET_ROOT}...")
    
    for split, (start, end) in splits.items():
        print(f"Processing {split} set...")
        
        # Process Real
        for video_path in tqdm(sampled_real[start:end], desc=f"{split}-real"):
            dest = os.path.join(TARGET_ROOT, split, 'real', video_path.name)
            shutil.copy2(video_path, dest)
            
        # Process Fake
        for video_path in tqdm(sampled_fake[start:end], desc=f"{split}-fake"):
            dest = os.path.join(TARGET_ROOT, split, 'fake', video_path.name)
            shutil.copy2(video_path, dest)
            
    print("Dataset creation complete!")
    print(f"Structure created at: {TARGET_ROOT}")
    print(f"Total Videos: {TOTAL_SAMPLES} | Train: {n_train*2} | Val: {n_val*2} | Test: {(samples_per_class - (n_train+n_val))*2}")

if __name__ == "__main__":
    main()
