import torch
import torch.nn as nn
import torchvision.models as models

class VideoBranch(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=2):
        super(VideoBranch, self).__init__()
        # Load Pretrained ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.cnn_out_dim = 512 

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = hidden_dim * 2 # 512

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn_backbone(c_in)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        return lstm_out # (B, T, 512)

class AudioBranch(nn.Module):
    def __init__(self, output_dim=512):
        super(AudioBranch, self).__init__()
        # Input: (Batch, 1, 128, 313) approx for 5s audio
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class HybridDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridDetector, self).__init__()
        
        self.video_branch = VideoBranch(hidden_dim=256)
        self.audio_branch = AudioBranch(output_dim=512)
        
        # Transformer
        # Input d_model = 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, video_frames, audio_spec):
        # Video: (B, T, 3, 224, 224) -> (B, T, 512)
        vid_seq = self.video_branch(video_frames)
        
        # We condense video to a single token representative? 
        # Or keep sequence? Let's reduce temporal dim by averaging for the token.
        vid_token = torch.mean(vid_seq, dim=1) # (B, 512)
        vid_token = vid_token.unsqueeze(1) # (B, 1, 512)
        
        # Audio: (B, 1, F, T) -> (B, 512)
        aud_token = self.audio_branch(audio_spec)
        aud_token = aud_token.unsqueeze(1) # (B, 1, 512)
        
        # Concat as sequence: [VideoToken, AudioToken]
        tokens = torch.cat((vid_token, aud_token), dim=1) # (B, 2, 512)
        
        # Transformer Logic
        trans_out = self.transformer(tokens) # (B, 2, 512)
        
        # Fuse final (Avg or Concat? Avg is simple)
        final_feat = torch.mean(trans_out, dim=1) # (B, 512)
        
        logits = self.classifier(final_feat)
        return logits
