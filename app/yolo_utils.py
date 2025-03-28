import cv2
import numpy as np
from ultralytics import YOLO
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def load_yolo_model():
    try:
        YOLO_MODEL_PATH = os.path.join(settings.BASE_DIR, 'app', 'yolov8n.pt')
        logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
        
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error("YOLO model file not found!")
            # Download the model if not found
            model = YOLO('yolov8n.pt')  # This will download if not present
            model.save(YOLO_MODEL_PATH)
            logger.info("Downloaded and saved YOLO model")
        else:
            model = YOLO(YOLO_MODEL_PATH)
        
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        raise

model = load_yolo_model()

def process_video_with_yolo(input_video_path, output_dir):
    try:
        logger.info(f"Starting YOLO processing for: {input_video_path}")
        
        # Verify input video exists
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
        
        # Prepare output paths
        video_name = os.path.basename(input_video_path)
        output_video_path = os.path.join(output_dir, video_name)
        frames_dir = os.path.join(settings.BASE_DIR, 'frames')
        
        # Open the video file
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {input_video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError("Could not initialize video writer")
        
        frame_count = 0
        detection_frames = []
        detection_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = model(frame, verbose=False)  # Disable ultralytics output
            
            # Filter only **people** (class ID = 0)
            person_boxes = []
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Check if class is "person"
                    person_boxes.append(box)
            
            # If people are detected
            if len(person_boxes) > 0:
                detection_count += 1
                
                # Draw only **person** detections on frame
                annotated_frame = frame.copy()
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    confidence = box.conf[0]  # Confidence score
                    
                    # Draw rectangle & label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ShopLifter: {confidence:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save every 10th frame with detections as sample
                if frame_count % 10 == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, annotated_frame)
                    detection_frames.append(os.path.join("frames", f"frame_{frame_count}.jpg"))
                
                # Write the annotated frame
                out.write(annotated_frame)
            else:
                # Write original frame if no detections
                out.write(frame)
            
            frame_count += 1
        
        logger.info(f"Processed {frame_count} frames with {detection_count} detections")
        
        # Release resources
        cap.release()
        out.release()
        
        if frame_count == 0:
            raise ValueError("No frames were processed from the video")
        
        return {
            'output_video': output_video_path,
            'detection_frames': detection_frames,
            'frame_count': frame_count,
            'detection_count': detection_count
        }
        
    except Exception as e:
        logger.error(f"Error in YOLO processing: {str(e)}")
        # Clean up partially created files
        if 'out' in locals() and out.isOpened():
            out.release()
        raise
