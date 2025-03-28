import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def classify_video(video_path, model, processor, num_frames=16):
    # Extract frames
    frames = extract_frames(video_path, num_frames)

    # Preprocess frames
    inputs = processor(frames, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].squeeze(1).to(device)  # Remove extra dimension and move to device

    # Run inference
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"])
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    # Map prediction to class name
    class_names = ["Non-Shoplifter", "Shoplifter"]
    predicted_class = class_names[preds[0]]

    return predicted_class