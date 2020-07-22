import cv2 as cv
from modules.face_detector.face_detector_module import FaceDetector
from modules.face_recognition.dlib_face_recognizer import recognize_face

import dlib
import glob
import numpy as np

def main():
    # initialise video input and Face Detector
    face_detector = FaceDetector('../modules/face_detector/data/face_detection_front.tflite')

    #intitalize face recognizer
    predictor_path = "../modules/face_recognition/data/shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "../modules/face_recognition/data/dlib_face_recognition_resnet_model_v1.dat"

    # Load all the models we need: a shape predictor
    # to find face landmarks so we can precisely localize the face, and finally the
    # face recognition model.

    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    # import predefined feature vector here
    # intitalize the generate_face_compare_fn with it

    faces_path = '../data/face_data/face_images/*.jpg'
    vector_save_path = '../data/face_data/face_encodings'

    for img_path in glob.glob(faces_path):
        person_name = img_path.split('/')[-1].replace(".jpg", '')
        print(f"processing: {img_path.split('/')[-1]}")

        img = cv.imread(img_path)

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        bounding_box_coordinates = face_detector.predict_face(img_rgb)

        if bounding_box_coordinates:

            x1, y1, x2, y2 = bounding_box_coordinates

            # draw bounding box around face
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # extract feature vector
            feature_vector = recognize_face(img_rgb, facerec, sp, bounding_box_coordinates, 100)
            print(f"Feature vector shape: {feature_vector.shape}")


            np.save(vector_save_path + "/" + person_name, feature_vector)
            cv.imwrite(vector_save_path + "/" + person_name + '.jpg', img)
            #cv.imshow(person_name, img)

if __name__ == "__main__":
    main()