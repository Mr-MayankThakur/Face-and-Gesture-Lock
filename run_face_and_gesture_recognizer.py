import time
import glob

import cv2 as cv
import numpy as np

# gesture detector and recognizer imports
from modules.hand_tracking.hand_tracker import HandTracker
from modules.gesture_recognizer.model import generate_gesture_score_fn
from modules.io_utils.threaded_cv_input import WebcamVideoStream

# face recognizer and detector imports
import dlib
from modules.face_detector.face_detector_module import FaceDetector
from modules.face_recognition.dlib_face_recognizer import recognize_face, generate_face_compare_fn

# region ---------------- Gesture Recognizer Setup -------------
gesture_path = './data/gesture_data/gesture_encodings/gesture1_1.npy'  # gesture file path (.npy)
saved_gesture = np.load(gesture_path, allow_pickle=True)
saved_gesture_name = gesture_path.split('/')[-1].replace(".npy", '')
get_gesture_score = generate_gesture_score_fn(saved_gesture)

palm_model_path = "./modules/hand_tracking/models/palm_detection.tflite"
landmark_model_path = "./modules/hand_tracking/models/hand_landmark.tflite"
anchors_path = "./modules/hand_tracking/data/anchors.csv"

detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)
# endregion ------------------------------------------------------

# region---------------- Face Recognizer Setup ---------------


def generate_face_embeddings(path):
    embedding = []
    names = []
    for vector in glob.glob(path):
        embedding.append(np.load(vector))
        names.append(vector.split('/')[-1].replace(".npy", ''))
    return np.array(embedding).T, names


face_detector = FaceDetector('./modules/face_detector/data/face_detection_front.tflite')

# intitalize face recognizer
predictor_path = "./modules/face_recognition/data/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./modules/face_recognition/data/dlib_face_recognition_resnet_model_v1.dat"

# Load all the models we need: a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.

sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# import predefined feature vector here
my_feature_vec, names = generate_face_embeddings('./data/face_data/face_encodings/*.npy')
# intitalize the generate_face_compare_fn with it
face_score = generate_face_compare_fn(my_feature_vec)

# endregion------------------------------------------------------


def find_face():
    bounding_box_coordinates = face_detector.predict_face(img_rgb)

    if bounding_box_coordinates:
        x1, y1, x2, y2 = bounding_box_coordinates

        # draw bounding box around face
        cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        current_feature_vector = recognize_face(img_rgb, facerec, sp, bounding_box_coordinates)

        scores = face_score(current_feature_vector[:, None])

        person_index = np.argmin(scores)

        if scores[person_index] > 0.5:
            return 0, "Unknown Person"
        else:
            return 1, f"Hello {names[person_index]}"
    else:
        return -1, "No Face Found"


def find_gesture():
    try:
        # kp is points list
        # box is box coordinates
        kp, box = detector(img_rgb)

        # do image drawing here
        for point in kp.astype(int):
            cv.circle(img, tuple(point), 5, (255, 0, 0), thickness=-1)

        # print(f"gesture score: {gesture_score(Y=kp, reflection=False)}")

        if get_gesture_score(Y=kp, reflection=False) < 0.1:
            return 1, f"Gesture Found: {saved_gesture_name}"
        else:
            return 0, "Wrong Gesture"

    except ValueError:
        # print("No Hands Found")
        return -1, "No Hands Found"

# initializing threaded video input
vs = WebcamVideoStream().start()

frame_cnt = 0
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = time.time()
while (True):
    img = vs.read()
    ret = img is not None

    img_height = img.shape[0]
    img_width = img.shape[1]

    frame_cnt += 1
    #print('-------- frame_cnt: ' + str(frame_cnt) + ' --------')

    if ret:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        face_status, face_message = find_face()
        gesture_status, gesture_message = find_gesture()

        print(f"Results: {face_message}, {gesture_message}")


    curr_time = time.time()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1

    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0

    #print(fps)

    cv.imshow('img', img)
    # vs.show('img', img)

    c = cv.waitKey(1) & 0xff

    if c == 27:
        break

vs.stop()
cv.destroyAllWindows()