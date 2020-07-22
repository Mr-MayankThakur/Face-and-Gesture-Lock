import glob

import cv2 as cv
import numpy as np

from modules.hand_tracking.hand_tracker import HandTracker

palm_model_path = "../modules/hand_tracking/models/palm_detection.tflite"
landmark_model_path = "../modules/hand_tracking/models/hand_landmark.tflite"
anchors_path = "../modules/hand_tracking/data/anchors.csv"

# use img_rgb
# box_shift determines
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)

gesture_folder_path = '../data/gesture_data/gesture_images'  # gesture images path
gesture_save_path = '../data/gesture_data/gesture_encodings'

# if verbose then the script will show images with discovered hand points
verbose = False

for file in glob.glob(gesture_folder_path + "/*.jpg"):
    file_name = file.split('/')[-1].replace(".jpg", '')

    img = cv.imread(file)

    img_height = img.shape[0]
    img_width = img.shape[1]

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    try:
        # kp is points list
        # box is box coordinates
        kp, box = detector(img_rgb)

        if verbose:
            # do image drawing here
            for point in kp.astype(int):
                cv.circle(img, tuple(point), 5, (255, 0, 0), thickness=-1)

        # save gesture
        np.save(f"{gesture_save_path}/{file_name}", kp)

        # save gesture image
        cv.imwrite(f"{gesture_save_path}/{file_name}.jpg", img)

    except ValueError:
        # print("No Hands Found")
        pass

    if verbose:
        cv.imshow('img', img)

        c = cv.waitKey(1) & 0xff

        if c == 27:
            break