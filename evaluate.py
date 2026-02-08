import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from model import HybridDetector
from train import DeepFakeDataset, BATCH_SIZE
import os

# Configuration
CHECKPOINT_PATH = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\checkpoints\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found. Please train the model first.")
        return

    # Load Data
    test_dataset = DeepFakeDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    model = HybridDetector(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for vid, aud, labels in test_loader:
            vid, aud = vid.to(DEVICE), aud.to(DEVICE)
            
            outputs = model(vid, aud)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability of Fake
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"AUC Score: {auc:.4f}")
    
    # Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    if auc > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
        print("Saved roc_curve.png")

if __name__ == "__main__":
    evaluate()
