# test_yolo.py
from ultralytics import YOLO
import cv2

def test_yolo():
    # Test with sample image
    model = YOLO('yolov8n.pt')
    results = model('https://ultralytics.com/images/bus.jpg')
    results[0].show()
    print("Image test successful!")
    
    # Test with sample video
    cap = cv2.VideoCapture(r"C:\Data Sets\ShopLift\Shop DataSet\shop lifters\shop_lifter_0.mp4") 
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        results[0].show()
        print("Video frame test successful!")
    else:
        print("Failed to read video frame")

if __name__ == '__main__':
    test_yolo()