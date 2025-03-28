import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import torch
import shutil
import signal
import atexit
from django.conf import settings
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from functions import classify_video
from .yolo_utils import process_video_with_yolo

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
try:
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-short")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-short", num_labels=2
    )
    model_path = os.path.join(settings.BASE_DIR, 'app', 'shoplifting_detector.pt')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def home(request):
    return render(request, 'home.html')

def detect_shoplifter(request, video_filename):
    """Handles shoplifting detection with classification and YOLO processing."""
    video_path = os.path.join(settings.MEDIA_ROOT, video_filename)
    video_url = f"{settings.MEDIA_URL}{video_filename}"

    prediction = classify_video(video_path, model, processor)

    if prediction == "Shoplifter":
        output_dir = settings.YOLO_ROOT  # Save processed videos in 'yolo' folder
        try:
            results = process_video_with_yolo(video_path, output_dir)
            if results.get('detection_count', 0) == 0:
                return render(request, 'result.html', {
                    'prediction': "Non-Shoplifter",
                    'video_url': video_url,
                })

            # Serve processed video from YOLO folder
            processed_video_url = f"{settings.YOLO_URL}{os.path.basename(results['output_video'])}"

            return render(request, 'detection.html', {
                'original_video': video_url,
                'processed_video': processed_video_url,
                'prediction': prediction
            })
        except Exception as yolo_error:
            print(f"YOLO processing failed: {yolo_error}")
            return render(request, 'result.html', {
                'prediction': prediction,
                'video_url': video_url,
                'message': "Shoplifting suspected but detection failed"
            })

    return render(request, 'result.html', {
        'prediction': prediction,
        'video_url': video_url
    })


def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        fs = FileSystemStorage()
        try:
            video_file = request.FILES['video']
            filename = fs.save(video_file.name, video_file)
            return detect_shoplifter(request, filename)
        except Exception as e:
            print(f"Error processing video: {e}")
            return render(request, 'error.html', {
                'error': str(e),
                'message': "Failed to process video"
            })
    return render(request, 'upload.html')

MEDIA_PATH = os.path.join(settings.BASE_DIR, 'media')
YOLO_PATH = os.path.join(settings.BASE_DIR, 'yolo')  
FRAMES_PATH = os.path.join(settings.BASE_DIR, 'frames') 

def cleanup_folder(folder_path):
    """Deletes all contents inside a folder but keeps the folder itself."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        print(f"🧹 {folder_path} contents deleted.")

def cleanup_media():
    cleanup_folder(MEDIA_PATH)
    cleanup_folder(YOLO_PATH)
    cleanup_folder(FRAMES_PATH)

atexit.register(cleanup_media)

def handle_exit_signal(signum, frame):
    cleanup_media()
    print("Server stopped. Media folder contents removed.")
    exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)