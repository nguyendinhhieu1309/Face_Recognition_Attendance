
# 📸 Face Recognition Attendance System

An innovative Face Recognition Attendance System that utilizes facial recognition to automate the attendance process in educational or corporate settings. Built using advanced technologies like OpenCV, MTCNN, and FaceNet, this system can capture, train, and recognize faces, making attendance more efficient and accurate.

## 🎯 Features
- **Student Image Capture**: Automatically captures images of students through an IP camera.
- **Face Detection**: Uses MTCNN to detect faces in real-time.
- **Face Recognition**: Implements FaceNet for embedding facial features and SVM for classification.
- **Attendance Logging**: Recognized faces are logged into an Excel sheet with the timestamp.
- **GUI Interface**: Easy-to-use interface built with Tkinter.

## 🛠️ Technologies and Libraries Used
- ![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
- ![MTCNN](https://img.shields.io/badge/MTCNN-v1.0.0-orange.svg)
- ![FaceNet](https://img.shields.io/badge/FaceNet-Keras-orange.svg)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-yellow.svg)
- ![Pandas](https://img.shields.io/badge/pandas-1.3.3-brightgreen.svg)
- ![OpenPyXL](https://img.shields.io/badge/OpenPyXL-3.0.7-blue.svg)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```
### 2. Install dependencies
Make sure you have Python 3.9 or higher installed. Then, install the required libraries using pip:

```bash
pip install opencv-python opencv-python-headless mtcnn keras-facenet tensorflow scikit-learn pandas openpyxl
```

### 3. Set up IP Camera
Ensure that you have an IP camera running, and update the camera URL in the code:

```python
url = 'http://<your-camera-ip>:<port>/video'
```

### 4. Run the Application
To start the application, simply run:

```bash
python main.py
```

## 📂 Project Structure
```
Face_Recognition_Attendance_System/
│
├── data_utils.py            # Functions to handle student image capturing
├── train.py                 # Script to train the face recognition model
├── recognize.py             # Script to recognize faces in real-time
├── main.py                  # Main Tkinter GUI application
├── students.csv             # Database of student IDs and names
├── attendance.xlsx          # Excel sheet where attendance is logged
├── models/                  # Directory containing saved models
│   ├── weights/             # Directory for SVM model and encoders
│   └── src/                 # Source directory for fonts, etc.
└── README.md                # Documentation
```

## 🤖 Model Overview
The core of this system revolves around **FaceNet**, which is used to generate embeddings (numerical representations) of faces. These embeddings are then fed into a **Support Vector Machine (SVM)** classifier to distinguish between different students based on their faces.

### Workflow:
1. **Face Detection**: Using MTCNN, faces are detected in real-time from a video stream.
2. **Face Embedding**: Detected faces are passed through the FaceNet model to get 128-dimension embeddings.
3. **Classification**: The SVM classifier uses these embeddings to recognize the student.
4. **Logging Attendance**: If a student is recognized, their attendance is marked in an Excel sheet with the current timestamp.

## 🖥️ GUI Interface
The system includes a simple GUI built with **Tkinter**, allowing easy interaction:
- **Add Student**: Enter the student ID and name to add a new student to the database.
- **Train Model**: Train the face recognition model with newly captured images.
- **Track Face**: Start face recognition and log attendance automatically.

## 📊 Results
- The system can achieve high accuracy (95-98%) in recognizing students, depending on the quality of the images and lighting conditions.
- The model can distinguish between faces even in challenging conditions like low light or partially obstructed faces.

## 📸 Demo and Results
Here are some sample images and demo results from the Face Recognition Attendance System:

- **Image 1**: A screenshot of real-time face detection and recognition with the bounding box around the face and the predicted student ID.
- **Image 2**: The attendance Excel sheet showing student IDs, names, and the timestamp of recognition.

![image](https://github.com/user-attachments/assets/5d1ce064-80a1-4c1e-a735-c92bc1f6518f)

![image](https://github.com/user-attachments/assets/875d07cf-2ffc-42b3-9167-f4b7a922b28a)


The demo shows the robustness of the system in recognizing multiple faces with high accuracy in real-time.

## 🤝 Contributing
Contributions are welcome! Here's how you can help:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

Please feel free to open issues if you find any bugs or have suggestions for improvements.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author
Developed by **Your Name** - [LinkedIn](https://www.linkedin.com/in/nguyen-dinh-hieu-818778303/)

---

_If you found this project helpful, please give it a ⭐ on [GitHub](https://github.com/nguyendinhhieu1309/Face_Recognition_Attendance.git)!_
