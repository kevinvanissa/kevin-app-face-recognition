# Family Face Recognition Tracking System

## Overview

The Family Face Recognition Tracking System is a Python-based application designed to detect and track faces in real-time using a webcam. This application leverages a foundational face recognition model trained on images of my family members. It identifies known individuals and saves any unknown faces to the filesystem for later review. The system utilizes the `face_recognition` library for facial recognition, `OpenCV` for video processing, and various other libraries for performance and evaluation.


## Features

- **Real-time Face Detection**: Detect faces using a webcam feed.
- **Face Recognition**: Recognize known faces based on pre-trained encodings.
- **Unknown Face Detection**: Save images of unknown faces for further analysis.
- **Model Validation**: Evaluate the performance of the recognition model using a validation dataset.

## Prerequisites
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

