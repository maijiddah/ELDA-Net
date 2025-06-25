
# elda_net_experiment.py

"""
ELDA-Net: Edge-Lightweight Detection and Adaptation Network
Full experiment pipeline: training, evaluation, and lane detection.
This code is prepared for publication and GitHub release.
"""

import os
import cv2
import torch
import numpy as np
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from model.unet import UNet
from utils.dataset import LaneDataset
from utils.preprocessing import preprocess_image
from utils.visualization import overlay_lanes
from utils.adaptive_estimator import AdaptiveLaneEstimator
from utils.metrics import compute_metrics

# Load Configuration from external YAML file
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

def get_data_paths():
    if CONFIG['dataset'] == 'TuSimple':
        return CONFIG['tusimple']['image_dir'], CONFIG['tusimple']['label_dir']
    elif CONFIG['dataset'] == 'CULane':
        return CONFIG['culane']['image_dir'], CONFIG['culane']['label_dir']
    else:
        raise ValueError("Unknown dataset selected")

def train():
    print("[INFO] Starting training process...")
    image_dir, label_dir = get_data_paths()
    dataset = LaneDataset(image_dir, label_dir, CONFIG['input_size'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Using a slightly customized U-Net architecture adapted for lane detection
    model = UNet(in_channels=3, out_channels=1).to(CONFIG['device'])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        running_loss = 0.0
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Loss: {running_loss/len(dataloader):.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = CONFIG['model_save_path'].replace('.pth', f'_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")

def evaluate():
    print("[INFO] Evaluating model...")
    image_dir, label_dir = get_data_paths()
    dataset = LaneDataset(image_dir, label_dir, CONFIG['input_size'], train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=CONFIG['device']))
    model.eval()

    metrics = []
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
            outputs = model(imgs)
            preds = torch.sigmoid(outputs) > 0.5
            metrics.append(compute_metrics(preds.cpu(), masks.cpu()))

    avg_metrics = np.mean(metrics, axis=0)
    print(f"[RESULTS] F1 Score: {avg_metrics[0]:.4f}, IoU: {avg_metrics[1]:.4f}")

def infer_on_video(video_path, output_path):
    print(f"[INFO] Running inference on {video_path}...")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    out = None
    estimator = AdaptiveLaneEstimator()

    model = UNet(in_channels=3, out_channels=1).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=CONFIG['device']))
    model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_image(frame, CONFIG['input_size'])
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(CONFIG['device'])

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5

        adapted_mask = estimator.refine(pred_mask)
        overlay = overlay_lanes(frame, adapted_mask)

        if out is None:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                  (overlay.shape[1], overlay.shape[0]))

        out.write(overlay)

    cap.release()
    out.release()
    print(f"[INFO] Output saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run ELDA-Net experiment pipeline')
    parser.add_argument('--mode', choices=['train', 'eval', 'infer'], default='train', help='Mode to run: train, eval, or infer')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--dataset', type=str, choices=['TuSimple', 'CULane'], help='Dataset to use')
    args = parser.parse_args()

    if args.dataset:
        CONFIG['dataset'] = args.dataset

    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate()
    elif args.mode == 'infer':
        if args.video and args.output:
            infer_on_video(args.video, args.output)
        else:
            print("[ERROR] --video and --output must be specified in infer mode.")
    else:
        print("[ERROR] Invalid mode. Use --mode with 'train', 'eval', or 'infer'.")
