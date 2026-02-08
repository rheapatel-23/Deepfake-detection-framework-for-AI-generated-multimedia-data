# Deepfake Detection Framework for AI-Generated Multimedia Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A powerful, multimodal deepfake detection system that utilizes a **Hybrid CNN-LSTM-Transformer** architecture to detect manipulations in both video and audio streams.

## 🚀 Overview
Deepfakes are becoming increasingly sophisticated. This project addresses the challenge by analyzing:
- **Spatial Features**: Using ResNet18 for frame-level analysis.
- **Temporal Features**: Using Bi-Directional LSTM to detect anomalies across time.
- **Audio Features**: Using 2D CNN (ResNet) to analyze Mel-Spectrograms.
- **Cross-Modal Fusion**: Using a Transformer to correlate audio and visual cues.

## ✨ Key Features
- **Multimodal Detection**: Analyzes both sound and video.
- **Hybrid Architecture**: Combines ResNet (CNN), LSTM, and Transformers.
- **Explainability**: Frame Importance Graph shows which parts of the video were most suspicious.
- **Interactive Dashboard**: Easy-to-use Streamlit UI for drag-and-drop analysis.
- **Detailed Reporting**: Automated metric calculation (Accuracy, ROC, Confusion Matrix).

## 📊 Performance
- **Test Accuracy**: 87.67%
- **AUC Score**: 0.96

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/rheapatel-23/Deepfake-detection-framework-for-AI-generated-multimedia-data.git
cd Deepfake-detection-framework-for-AI-generated-multimedia-data

# Install dependencies
pip install -r requirements.txt
```
*(Note: Create a `requirements.txt` or install: `torch torchvision opencv-python librosa pandas tqdm sklearn matplotlib seaborn streamlit plotly python-docx`)*

## 📖 Usage

### 🖥️ Start the Dashboard
```bash
streamlit run app.py
```

### 🧠 Training
```bash
python train.py
```

### 🧪 Evaluation
```bash
python evaluate.py
```

### 🔍 Single Video Inference
```bash
python inference.py "path/to/video.mp4"
```

## 🎥 Demo Videos
For quick testing, I have included sample videos in the `demos/` directory:
- `demos/sample_real.mp4`: A verified real video.
- `demos/sample_fake.mp4`: A deepfake video with audio/visual manipulation.

Try running:
```bash
python inference.py "demos/sample_fake.mp4"
```

## 📂 Project Structure
- `model.py`: Hybrid architecture definition.
- `data_loader.py`: Dataset sampling and structure logic.
- `preprocess.py`: Feature extraction (Faces & Audio).
- `train.py`: Training pipeline.
- `app.py`: Streamlit UI.
- `inference.py`: End-to-end prediction logic.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
