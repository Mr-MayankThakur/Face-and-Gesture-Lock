from modules.hand_tracking.hand_tracker import HandTracker
from modules.gesture_recognizer.model import generate_gesture_score_fn
from modules.io_utils.threaded_cv_input import WebcamVideoStream

import cv2 as cv
import numpy as np
import time

# gesture file path (.npy)
gesture_path = '../data/gesture_data/gesture_encodings/gesture1_1.npy'
saved_gesture = np.load(gesture_path, allow_pickle=True)

gesture_score = generate_gesture_score_fn(saved_gesture)

# initializing threaded video input

palm_model_path = "../modules/hand_tracking/models/palm_detection.tflite"
landmark_model_path = "../modules/hand_tracking/models/hand_landmark.tflite"
anchors_path = "../modules/hand_tracking/data/anchors.csv"


# use img_rgb
# box_shift determines
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)

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

        try:
            # kp is points list
            # box is box coordinates
            kp, box = detector(img_rgb)

            # do image drawing here
            for point in kp.astype(int):
                cv.circle(img, tuple(point), 5, (255,0,0), thickness=-1)

            #print(f"gesture score: {gesture_score(Y=kp, reflection=False)}")

            if gesture_score(Y=kp, reflection=False) < 0.1:
                print("Gesture Found")
            else:
                print("Wrong Gesture")

        except ValueError:
            #print("No Hands Found")
            pass


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