# **Anti-Theft Detection System**  
### *Advanced AI-Powered Shoplifting Detection for Retail Security*  

## **Overview**  
The **Anti-Theft Detection System** is a cutting-edge AI-powered solution designed to enhance retail security by detecting potential shoplifting behavior in real time. The system utilizes computer vision and deep learning to analyze video footage and classify individuals as either **"Shoplifter"** or **"Non-Shoplifter"** with high accuracy.  

By integrating **YOLOv8 for object detection** and a **custom-trained deep learning model**, the system provides an efficient and automated method for loss prevention in retail stores.  

## **Project Structure**  

```
Anti-Theft-Detection/
│── Anti-Theft/
│   ├── app/
│   │   ├── migrations/                # Database migrations
│   │   ├── templates/                 # Frontend HTML templates
│   │   │   ├── base.html
│   │   │   ├── detection.html
│   │   │   ├── error.html
│   │   │   ├── home.html
│   │   │   ├── result.html
│   │   │   ├── upload.html
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py                   # Django models
│   │   ├── tests.py
│   │   ├── urls.py                      # URL routing
│   │   ├── views.py                     # Backend logic for detection
│   │   ├── yolo_utils.py                 # Utility functions for YOLO
│   │   ├── shoplifting_detector.pt       # Trained AI model for classification
│   │   ├── yolov8n.pt                    # YOLOv8 model for object detection
│── Base/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│── frames/                              # Extracted frames from videos
│── media/                               # Uploaded video files
│── yolo/
│   ├── db.sqlite3                        # Database
│   ├── functions.py                      # Video processing functions
│   ├── install.ipynb                      # Setup and installation guide
│── manage.py                              # Django management script
```

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/ehab1441/ShopLift-Detection
cd Anti-Theft-Detection
```

### **2. Create & Activate a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```
### 4️. **Download & Install Model Files**  
Before running the project, download the pre-trained AI models (**YOLOv8** & **Shoplifting Detector**) from the following links:

🔗 **YOLOv8 Model** (Object Detection)  
📥 [Download YOLOv8](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)  

🔗 **Shoplifting Detection Model**  
📥 [Download Shoplifting Detector](https://drive.google.com/drive/folders/1V1s3CV8UkhwJLpZ9xYu_mbFOkD_GqwZn?usp=sharing)  

After downloading, move the files to the `app/` directory:

```bash
mv yolov8n.pt Anti-Theft/app/
mv shoplifting_detector.pt Anti-Theft/app/

python manage.py migrate
python manage.py runserver
```

### **5. Access the Web App**  
Open your browser and go to:  
🔗 **http://127.0.0.1:8000/**  

---

## **How It Works**  
1. **Upload Video** - The user uploads surveillance footage via the web interface.  
2. **Frame Extraction** - The system extracts frames from the video using OpenCV.  
3. **AI Processing** - The frames are analyzed by the **shoplifting_detector.pt** model.  
4. **Object Detection (YOLO)** - Additional object detection is performed using **YOLOv8**.  
5. **Classification** - The system classifies whether the subject is a **Shoplifter** or **Non-Shoplifter**.  
6. **Results** - The detection results are displayed on the web interface.  

---

## **Model Performance**  

| Metric          | Value  |
|----------------|--------|
| **Train Loss**  | 0.0550  |
| **Train Accuracy** | 97.30%  |
| **Validation Loss** | 0.0291  |
| **Validation Accuracy** | 98.44%  |

🚀 **High accuracy ensures reliable detection for real-world applications!**  

---

## **Technologies Used**  
🖥️ **Backend** - Django (Python)  
🎨 **Frontend** - HTML, CSS, Bootstrap  
🤖 **Machine Learning** - PyTorch, OpenCV, Transformers  
🎯 **Object Detection** - YOLOv8  

---

## **Future Enhancements**  
✅ Improve real-time detection speed  
✅ Expand dataset for better generalization  
✅ Add an alert system for instant notifications  
✅ Implement real-time tracking for multiple individuals  

---

## **Contributors**  
👨‍💻 **Mohamed Ehab** – Lead Developer  

💡 *Want to contribute? Feel free to fork the repo and submit a pull request!*  


