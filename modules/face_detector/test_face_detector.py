import cv2 as cv
from face_detector_module import FaceDetector
import time

def main():
    # initialise video input and Face Detector
    face_detector = FaceDetector('data/face_detection_front.tflite')

    # capture = cv2.VideoCapture('./videoplayback_1.mp4')
    capture = cv.VideoCapture(0)
    frame_cnt = 0
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    while (True):
        ret, img = capture.read()

        # img = cv2.imread('./test_image.jpg')


        img_height = img.shape[0]
        img_width = img.shape[1]

        frame_cnt += 1
        print('-------- frame_cnt: ' + str(frame_cnt) + ' --------')
        if ret == True:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            bounding_box_coordinates = face_detector.predict_face(img_rgb)

            if bounding_box_coordinates:

                x1, y1, x2, y2 = bounding_box_coordinates

                # draw bounding box around face
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1

            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            print(fps)

            cv.imshow('img', img)

            c = cv.waitKey(1) & 0xff

            if c == 27:
                break

if __name__ == "__main__":
    main()