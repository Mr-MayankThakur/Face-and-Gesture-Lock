# todo redo this file with new funcions
#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough.
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy


import dlib

import time

predictor_path = "data/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "data/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "sample_images_1/"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

pred = []
descriptors = []
# --------------------------
import cv2 as cv
import numpy as np

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

        preprocess_start_time = time.time()

        # input shape
        input_width = min(img_rgb.shape)
        input_height = min(img_rgb.shape)

        # resize
        input_data = cv.resize(img_rgb, (input_width, input_height)).astype(np.float32)

        # preprocess
        # input_data = (input_data)
        input_data = ((input_data - 127.5) / 127.5)
        # input_data = ((input_data)/255)
        input_data = np.expand_dims(input_data, axis=0)
        preprocess_end_time = time.time()

        # -----------------------------
        # Now process all the images

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        detection_start_time = time.time()

        dets = detector(img, 1)
        # print("Number of faces detected: {}".format(len(dets)))

        detection_end_time = time.time()

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            face_descriptor_start = time.time()
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.

            # post processing

            x1 = shape.rect.left()
            x2 = shape.rect.right()
            y1 = shape.rect.top()
            y2 = shape.rect.bottom()
            print("x1: {:}, y1: {:}\nx2: {:}, y2: {:}".format(x1, y1, x2, y2))

            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people. Here we just print
            # the vector to the screen.

            face_descriptor = facerec.compute_face_descriptor(img, shape)

            # print(face_descriptor)
            #pred.append(face_descriptor)
            face_descriptor_end = time.time()
            # It should also be noted that you can also call this function like this:
            #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100, 0.25)
            # The version of the call without the 100 gets 99.13% accuracy on LFW
            # while the version with 100 gets 99.38%.  However, the 100 makes the
            # call 100x slower to execute, so choose whatever version you like.  To
            # explain a little, the 3rd argument tells the code how many times to
            # jitter/resample the image.  When you set it to 100 it executes the
            # face descriptor extraction 100 times on slightly modified versions of
            # the face and returns the average result.  You could also pick a more
            # middle value, such as 10, which is only 10x slower but still gets an
            # LFW accuracy of 99.3%.
            # 4th value (0.25) is padding around the face. If padding == 0 then the chip will
            # be closely cropped around the face. Setting larger padding values will result a looser cropping.
            # In particular, a padding of 0.5 would double the width of the cropped area, a value of 1.
            # would triple it, and so forth.

            # There is another overload of compute_face_descriptor that can take
            # as an input an aligned image.
            #
            # Note that it is important to generate the aligned image as
            # dlib.get_face_chip would do it i.e. the size must be 150x150,
            # centered and scaled.
            #
            # Here is a sample usage of that

            # print("Computing descriptor on aligned image ..")

            # Let's generate the aligned image using get_face_chip
            #face_chip = dlib.get_face_chip(img, shape)

            # Now we simply pass this chip (aligned image) to the api
            #face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
            # print(face_descriptor_from_prealigned_image)
            #descriptors.append(face_descriptor_from_prealigned_image)

            print(f"preprocess time = {preprocess_end_time - preprocess_start_time}")
            print(f"detection time = {detection_end_time - detection_start_time}")
            print(f"face_descripion = {face_descriptor_end - face_descriptor_start}")

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
            # print("waiting for next image")
