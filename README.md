# Family Face Recognition Tracking System

## Overview

The Family Face Recognition Tracking System is a Python-based application designed to detect and track faces in real-time using a webcam. This application leverages a foundational face recognition model trained on images of my family members. It identifies known individuals and saves any unknown faces to the filesystem for later review. The system utilizes the `face_recognition` library for facial recognition, `OpenCV` for video processing, and various other libraries for performance and evaluation.


## Features

- **Real-time Face Detection**: Detect faces using a webcam feed.
- **Face Recognition**: Recognize known faces based on pre-trained encodings.
- **Unknown Face Detection**: Save images of unknown faces for further analysis.
- **Model Validation**: Evaluate the performance of the recognition model using a validation dataset.

## Prerequisites
Clone repository:
```bash
git clone https://github.com/kevinvanissa/kevin-app-face-recognition.git
```


Create virtual environment:
```bash
python -m venv myenv
```
Ensure you have the following libraries installed:
- `face_recognition`
- `opencv-python`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install using the `requirements.txt`:
```bash
pip install -r requirements.txt
```
### OR
You can install these using pip:
```bash
pip install face_recognition opencv-python numpy matplotlib seaborn scikit-learn
```


## Directory Structure

- encodings/: Directory to store face encodings.
- training/: Directory containing images for training the model.
- validation/: Directory containing images for validating the model.
- unknown_frames/: Directory where unknown face images are saved.

**Important** : If these folders do not exist, they will be created when the program starts. However, there are some structures inside the `training/` and `validation/` folders that must be maintained for the application to work correctly. In the training folder, use subfolders with the correct names of the persons you want the application to know. The validation folder should have one face in each image (multiple images can have the same face) and the name of the person that matches the name of the folder in the training folder. I used multiple images with the same person. You should use an underscore with a number for pictures with the same person. For example, If the name of the person is kevin, then in the training folder, I will use a folder by the name of kevin and store multiple pictures of kevin. The name of these images can be anything. In the validation folder, I will have multiple images of kevin, so I will name the images as follows: kevin_1.jpg, kevin_2.jpg ...

I will include example folders in this repository (training.example, validation.example). 

## How to Use
### 1. Training the Model

To train the model using images from the `training/` folder, run:


```bash
python main.py --train
```

This command will generate and save face encodings in the `encodings/` directory.

### 2. Validating the Model


To validate the trained model using images from the `validation/` folder, run:

```bash
python main.py --validate
```
This will compute and display the accuracy, recall, and confusion matrix for the model.

### 3. Running the Application

To start the face recognition application using your default webcam, run:

```bash
python main.py --run
```
The application will display a live feed from the webcam, draw bounding boxes around recognized faces, and save images of unknown faces to the unknown_frames/ directory.

## Code Overview
- `encode_faces(model, encode_file_name)`: Encodes faces from images in the training/ directory and saves them to a file.
- `_save_frame()`: Saves frames of unknown faces to the filesystem.
- `_match_face(video_face_encodings, registered_face_encodings, frame)`: Matches faces in the current frame with registered encodings and adds unknown faces to a queue for saving.
- `_draw_face(face_locations, face_names, frame)`: Draws bounding boxes and labels on recognized faces in the frame.
- `validate_model()`: Validates the model by comparing predicted and true face labels from the validation/ directory.
- `run_app()`: Captures live video from the webcam, processes frames, and displays recognized faces.

