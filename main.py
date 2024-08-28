import os
import sys
import  face_recognition
import pdb
import pickle
import cv2
from collections import Counter
import argparse
import threading
import queue

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


ENCODING_FOLDER = os.path.join(BASE_DIR, "encodings")
TRAINING_FOLDER = os.path.join(BASE_DIR, "training")
VALIDATION_FOLDER = os.path.join(BASE_DIR, "validation")
UNKNOWN_FRAMES_FOLDER = os.path.join(BASE_DIR, "unknown_frames")

RECTANGLE_COLOR = "green"
TEXT_COLOR = "white"
MODEL_CHOICE = "hog"
ENCODED_FILE = "encodings.pkl"

if not os.path.exists(ENCODING_FOLDER):
    os.mkdir(ENCODING_FOLDER)

if not os.path.exists(TRAINING_FOLDER):
    os.mkdir(TRAINING_FOLDER)

if not os.path.exists(VALIDATION_FOLDER):
    os.mkdir(VALIDATION_FOLDER)

if not os.path.exists(UNKNOWN_FRAMES_FOLDER):
    os.mkdir(UNKNOWN_FRAMES_FOLDER)


# Thread-safe queue to hold frames for saving
frame_queue = queue.Queue()


def encode_faces(model=MODEL_CHOICE, encode_file_name = ENCODED_FILE):
    
    names = []
    encodings = []

    for dirpath, dirnames, filenames in os.walk(TRAINING_FOLDER):
        if dirpath == os.path.join(BASE_DIR, TRAINING_FOLDER):
            continue

        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            parent_dir_name = os.path.basename(dirpath)

            #load image
            image = face_recognition.load_image_file(image_path)

            # Return list of bouding boxes
            face_bounded_boxes= face_recognition.face_locations(image, model=model)
            # Return encodings (numerical representations of features of face) of face for the given image
            encoded_faces = face_recognition.face_encodings(image, face_bounded_boxes)

            # add each name and encoding to a list
            for encoding in encoded_faces:
                names.append(parent_dir_name)
                encodings.append(encoding)
            # pdb.set_trace()
            # create dictionary of names and encodings and save to filesytem
            encodings_dict = {"names": names, "encodings": encodings}
            with open(os.path.join(ENCODING_FOLDER, encode_file_name), "wb") as file:
                pickle.dump(encodings_dict, file)


def _save_frame():
    while True:
        unknown_frame = frame_queue.get()
        #Frame will be none at the end of the program as this is manually set
        if unknown_frame is None:
            break
        #TODO: use timestamps to save image
        cv2.imwrite(os.path.join(UNKNOWN_FRAMES_FOLDER, 'unknown.jpg'), unknown_frame)
        frame_queue.task_done()



def _match_face(video_face_encodings, registered_face_encodings, frame=None):
    face_names = []
    for face_encodings in video_face_encodings:
        boolean_matches = face_recognition.compare_faces(
                registered_face_encodings["encodings"], face_encodings
                )
        name = "Unknown"
        face_distances = face_recognition.face_distance(registered_face_encodings["encodings"], face_encodings)
        top_match_index = np.argmin(face_distances)
        if boolean_matches[top_match_index]:
            name = registered_face_encodings["names"][top_match_index]
        # if face is uknown then put frame in queue to be saved
        if name == "Unknown":
            print("Unknown face detected sending to queue to be processed")
            frame_queue.put(frame)
        face_names.append(name)

    return face_names

def _draw_face(face_locations, face_names, frame):

    #for (top, right, bottom, left), name in zip(face_locations, face_names):
    #locations is a tuple
    for locations, name in zip(face_locations, face_names):
        # remember we resized the frame for faster processing. So now we have to scale back the locations before drawing

        top, right, bottom, left = map(lambda x: x*4, locations)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)


def validate_model():
    with open(os.path.join(ENCODING_FOLDER, ENCODED_FILE ), "rb") as file:
            registered_face_encodings = pickle.load(file)

    y_true = []
    y_pred = []


    for filename in os.listdir(VALIDATION_FOLDER):
        input_image_file = face_recognition.load_image_file(os.path.join(VALIDATION_FOLDER, filename))

        input_face_locations = face_recognition.face_locations(input_image_file, model=MODEL_CHOICE)
        input_face_encodings = face_recognition.face_encodings(input_image_file, input_face_locations)

        face_name = _match_face(input_face_encodings, registered_face_encodings)
        filename = filename.split("_")[0]
        y_true.append(filename)
        y_pred.append(face_name[0])


    # calculate overall accuracy. Accuracy  = TP + TN / Total
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy : {accuracy: .2f}')

    # calculate recall(True Positive Rate). = TP/TP + FN 
    # use average=micro because this is a multiclass problem. I use compute recall globally by aggregating similar classes
    recall = recall_score(y_true, y_pred, average='micro')
    print(f'Recall: {recall: .2f}')

    labels = y_true
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Facial Recognition')
    plt.show()



def run_app():
        # Starting the frame-saving thread
        thread = threading.Thread(target=_save_frame)
        thread.start()

        with open(os.path.join(ENCODING_FOLDER, ENCODED_FILE ), "rb") as file:
            registered_face_encodings = pickle.load(file)

        # variable to control how often to process
        handle_current_frame = True
        # Reference to webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Get a single frame of the video
            ret, frame = cap.read()


            if handle_current_frame:
                #Resize the frame. This speeds up processing
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

                # convert the image from OpenCV BGR format to face_recognition RGB format
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                #print("shape and type")
                #print(rgb_frame.shape)
                #print(type(rgb_frame))

                # find video faces locations and encondings for current frame
                video_face_locations = face_recognition.face_locations(rgb_frame, model=MODEL_CHOICE) 
                video_face_encodings = face_recognition.face_encodings(rgb_frame, video_face_locations)

                #print(video_face_locations)
                #print(video_face_encodings)
                #print("entering bouded box video_encodings zip now...")

                face_names = _match_face(video_face_encodings, registered_face_encodings, frame)
                _draw_face(video_face_locations, face_names, frame)

                """
                for bounding_box, video_encodings in zip(video_face_locations, video_face_encodings):
                    #check if this face matches
                    name = "Unknown"
                    boolean_matches = face_recognition.compare_faces(
                            registered_face_encodings["encodings"], video_encodings)

                    votes = Counter(
                            name
                            for match, name in 
                            zip(boolean_matches, registered_face_encodings["names"])
                            if match
                            )
                    print("Votes!!")
                    print(votes)
                    if votes:
                        name = votes.most_common(1)[0][0]

                    # Draw a box around the face
                    top, right, bottom, left =  bounding_box
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)

                """

            # switch handle_current_frame variable
            handle_current_frame = not handle_current_frame

            # show image
            cv2.imshow('My Faces', frame)

            # when 'q' is pressed application will break out of the while loop
            if cv2.waitKey(1) == ord('q'):
                break


        #Signal the saving thread to finish
        frame_queue.put(None)

        # wait for the saving thread to finish
        thread.join()


        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="App to recognize faces in webcam")
    parser.add_argument("--train", action="store_true", help="Train using the images in training folder")
    parser.add_argument("--validate", action="store_true", help="Validate the model using images in the validation folder")
    parser.add_argument("--run", action="store_true", help="Run the application using default webcam")

    # Parse all incoming arguments
    args = parser.parse_args() 

    #if not vars(args):
        #print("No arguments provided")
    #print(vars(args))

    if not (args.train or args.validate or args.run):
        print("No arguments were given. Use python main.py --help")
        sys.exit(1)

    if args.train:
        print("Start training model...")
        encode_faces()
        print("Finished training model!")

    if args.validate:
        print("Validating Model..")
        validate_model()
        print("Finished Validating Model")

    if args.run:
        run_app()

