import cv2 as cv
from modules.face_detector.face_detector_module import FaceDetector
from modules.face_recognition.dlib_face_recognizer import recognize_face, generate_face_compare_fn
from modules.io_utils.threaded_cv_input import WebcamVideoStream

import dlib
import time
import glob
import numpy as np


def generate_face_embeddings(path):
    embedding = []
    names = []
    for vector in glob.glob(path):
        embedding.append(np.load(vector))
        names.append(vector.split('/')[-1].replace(".npy", ''))
    return np.array(embedding).T, names


def main():
    # initialise video input and Face Detector
    face_detector = FaceDetector('modules/face_detector/data/face_detection_front.tflite')

    # intitalize face recognizer
    predictor_path = "modules/face_recognition/data/shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "modules/face_recognition/data/dlib_face_recognition_resnet_model_v1.dat"

    # Load all the models we need: a shape predictor
    # to find face landmarks so we can precisely localize the face, and finally the
    # face recognition model.

    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    # import predefined feature vector here
    my_feature_vec, names = generate_face_embeddings('modules/temp/vectors/*.npy')
    # intitalize the generate_face_compare_fn with it
    face_score = generate_face_compare_fn(my_feature_vec)

    # capture = cv2.VideoCapture('./videoplayback_1.mp4')
    #capture = cv.VideoCapture(0)
    # parallel threaded capture
    vs = WebcamVideoStream().start()

    frame_cnt = 0
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    while (True):
        #ret, img = capture.read()
        img = vs.read()
        ret = True

        # img = cv2.imread('./test_image.jpg')


        img_height = img.shape[0]
        img_width = img.shape[1]

        frame_cnt += 1
        #print('-------- frame_cnt: ' + str(frame_cnt) + ' --------')
        if ret == True:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            bounding_box_coordinates = face_detector.predict_face(img_rgb)

            if bounding_box_coordinates:
                x1, y1, x2, y2 = bounding_box_coordinates

                # draw bounding box around face
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                current_feature_vector = recognize_face(img_rgb, facerec, sp, bounding_box_coordinates)

                scores = face_score(current_feature_vector[:, None])

                person_index = np.argmin(scores)

                if scores[person_index] > 0.5:
                    print("Unknown Person")
                else:
                    print(f"Hello {names[person_index]}")
            else:
                print("No Face Found")


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
            #vs.show('img', img)

            c = cv.waitKey(1) & 0xff

            if c == 27:
                break
    vs.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

# we can define multiple faces as embedding matrix
# one with the lowest distance will be the one
# if distance of all is more than a threshold then the face is of new person
