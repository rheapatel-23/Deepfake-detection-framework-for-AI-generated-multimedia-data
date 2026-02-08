# Deepfake Detection Framework - Project Report

## 1. Project Overview
We have built a **Hybrid Deepfake Detection Framework** capable of identifying manipulations in multimedia content. The system leverages a multimodal architecture that fuses:
-   **Video Branch**: Uses **ResNet18** for spatial feature extraction and **Bi-Directional LSTM** for temporal modeling.
-   **Audio Branch**: Uses a **2D CNN** (ResNet-style) to analyze Mel-Spectrograms.
-   **Fusion Layer**: Combines both streams to make a final decision.

## 2. Dataset & Training
-   **Dataset**: Sampled from `FakeAVCeleb_v1.2`.
-   **Training Size**: Balanced subset of ~600 videos.
-   **Preprocessing**:
    -   **Faces**: Extracted ~10 frames per video using Haar Cascades.
    -   **Audio**: Converted to Mel-Spectrograms (128 mel bands).

## 3. Evaluation Results
The model was evaluated on a held-out test set (unseen videos).

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **87.67%** | The model correctly classified 87.67% of test videos. |
| **AUC Score** | **0.9609** | Excellent discriminative ability between Real and Fake classes. |

### Visualizations
**(Images saved in project directory)**

-   **Confusion Matrix**: `confusion_matrix.png`
    -   Shows True Positives, True Negatives, False Positives, and False Negatives.
-   **ROC Curve**: `roc_curve.png`
    -   Illustrates the trade-off between sensitivity and specificity.

## 4. Method to Reproduce
### Training
To retrain the model from scratch:
```bash
python train.py
```
*Outputs `checkpoints/best_model.pth`*

### Evaluation
To generate metrics on the test set:
```bash
python evaluate.py
```

## 5. Next Steps
-   **Inference Script**: Create a script to accept a single raw `.mp4` file and output a prediction.
-   **Web Deployment**: Build a simple UI (e.g., Streamlit) for potential users to upload videos.
