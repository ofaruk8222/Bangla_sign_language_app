import os
import cv2
import json
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATA_DIR = r"D:\4-I\Projects\Thesis\Sample testing with english\ASL Dataset"
METADATA_FILE = os.path.join(DATA_DIR, "WLASL_v0.3.json")
OUTPUT_DIR = r"D:\4-I\Projects\Thesis\Sample testing with english\processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FRAMES = 60
BATCH_SIZE = 16
EPOCHS = 20
MAX_GLOSSES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def extract_and_preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_data = []
    with mp.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            pose = results.pose_landmarks.landmark if results.pose_landmarks else None
            l_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
            r_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
            
            def get_coords(landmarks, indices):
                return np.array([[landmarks[i].x, landmarks[i].y] for i in indices], dtype=np.float32)
            
            if pose and l_hand and r_hand:
                body_indices = [11, 12, 13, 14, 15, 16, 23, 24]
                body_pts = get_coords(pose, body_indices)
                hand_indices = list(range(21))
                l_hand_pts = get_coords(l_hand, hand_indices)
                r_hand_pts = get_coords(r_hand, hand_indices)
                
                neck = (body_pts[0] + body_pts[1]) / 2.0
                head = np.array([pose[0].x, pose[0].y], dtype=np.float32)
                scale = np.linalg.norm(head - neck)
                if scale == 0: scale = 1e-5
                
                body_pts_norm = (body_pts - neck) / scale
                l_hand_norm = l_hand_pts - l_hand_pts[0]
                r_hand_norm = r_hand_pts - r_hand_pts[0]
                
                frame_features = np.concatenate([body_pts_norm.flatten(), l_hand_norm.flatten(), r_hand_norm.flatten()])
                frames_data.append(frame_features)
            else:
                frames_data.append(np.zeros(100, dtype=np.float32))

    cap.release()
    if len(frames_data) == 0: return None
    frames_data = np.array(frames_data)
    
    body_seq = frames_data[:, :16]
    hands_seq = frames_data[:, 16:]
    
    for i in range(len(hands_seq)):
        if np.all(hands_seq[i] == 0):
            prev_idx, next_idx = i - 1, i + 1
            while prev_idx >= 0 and np.all(hands_seq[prev_idx] == 0): prev_idx -= 1
            while next_idx < len(hands_seq) and np.all(hands_seq[next_idx] == 0): next_idx += 1
            
            if prev_idx >= 0 and next_idx < len(hands_seq):
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                hands_seq[i] = (1 - alpha) * hands_seq[prev_idx] + alpha * hands_seq[next_idx]
            elif prev_idx >= 0: hands_seq[i] = hands_seq[prev_idx]
            elif next_idx < len(hands_seq): hands_seq[i] = hands_seq[next_idx]
                
    processed_frames = np.concatenate([body_seq, hands_seq], axis=1)
    
    orig_frames = len(processed_frames)
    if orig_frames == TARGET_FRAMES: return processed_frames
    elif orig_frames > TARGET_FRAMES:
        indices = np.linspace(0, orig_frames - 1, TARGET_FRAMES, dtype=int)
        return processed_frames[indices]
    else:
        padding = np.tile(processed_frames[-1], (TARGET_FRAMES - orig_frames, 1))
        return np.vstack([processed_frames, padding])

def process_dataset():
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)
        
    X_data, y_data = [], []
    class_to_idx = {}
    current_idx = 0
    
    print(f"Processing first {MAX_GLOSSES} classes...")
    for i, item in enumerate(data):
        if i >= MAX_GLOSSES:
            break
            
        gloss = item['gloss']
        if gloss not in class_to_idx:
            class_to_idx[gloss] = current_idx
            current_idx += 1
            
        for instance in item['instances']:
            video_id = str(instance['video_id'])
            video_path = os.path.join(DATA_DIR, f"{video_id}.mp4")
            
            if os.path.exists(video_path):
                features = extract_and_preprocess_video(video_path)
                if features is not None:
                    X_data.append(features)
                    y_data.append(class_to_idx[gloss])
                    
    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), np.array(X_data))
    np.save(os.path.join(OUTPUT_DIR, 'y_data.npy'), np.array(y_data))
    print(f"Saved {len(X_data)} samples.")
    return current_idx

class SLR_Dataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class SLR_LSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, num_classes=10):
        super(SLR_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def main():
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'X_data.npy')):
        num_classes = process_dataset()
    else:
        print("Processed data found. Skipping extraction.")
        num_classes = len(np.unique(np.load(os.path.join(OUTPUT_DIR, 'y_data.npy'))))

    dataset = SLR_Dataset(os.path.join(OUTPUT_DIR, 'X_data.npy'), os.path.join(OUTPUT_DIR, 'y_data.npy'))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SLR_LSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = 100 * correct / len(train_loader.dataset)
        
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == batch_y).sum().item()
                
        test_acc = 100 * test_correct / len(test_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")
    
    print("Pipeline Proven!")

if __name__ == "__main__":
    main()