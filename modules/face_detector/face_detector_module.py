from modules.face_detector.mediapipe_detector_port import *
import cv2 as cv

class FaceDetector():
    def __init__(self, model_path):
        ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128, min_scale=0.1484375, max_scale=0.75
                                                                     , anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
                                                                     , feature_map_width=[], feature_map_height=[]
                                                                     , strides=[8, 16, 16, 16], aspect_ratios=[1.0]
                                                                     , reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
                                                                     , fixed_anchor_size=True)

        self.anchors = gen_anchors(ssd_anchors_calculator_options)

        self.options = TfLiteTensorsToDetectionsCalculatorOptions(num_classes=1, num_boxes=896, num_coords=16
                                                             , keypoint_coord_offset=4, ignore_classes=[], score_clipping_thresh=100.0, min_score_thresh=0.75
                                                             , num_keypoints=6, num_values_per_keypoint=2, box_coord_offset=0
                                                             , x_scale=128.0, y_scale=128.0, w_scale=128.0, h_scale=128.0, apply_exponential_on_box_size=False
                                                             , reverse_output_order=True, sigmoid_score=True, flip_vertically=False)
        # blaze face model
        # https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite

        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # model input shape
        self.input_width = self.input_details[0]["shape"][1]
        self.input_height = self.input_details[0]["shape"][2]

        # model output values

    def _preprocess_input(self, image):
        input_data = cv.resize(image, (self.input_width, self.input_height)).astype(np.float32)
        # preprocess
        # input_data = (input_data)
        input_data = ((input_data - 127.5) / 127.5)
        # input_data = ((input_data)/255)
        return np.expand_dims(input_data, axis=0)

    def predict_face(self, image):
        """
        Returns the bounding box coordinates for face found in the image
        :param image: numpy array
        :return: [x1, y1, x2, y2]
            bounding box coordinates
        """
        # preprocess image
        input_data = self._preprocess_input(image)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        regressors = self.interpreter.get_tensor(self.output_details[0]["index"])
        classificators = self.interpreter.get_tensor(self.output_details[1]["index"])
        return self._post_processing(regressors, classificators, image.shape[1], image.shape[0])  # bounding box coordinates

    def _post_processing(self, regressors, classificators, img_width, img_height):
        raw_boxes = np.reshape(regressors, int(regressors.shape[0] * regressors.shape[1] * regressors.shape[2]))
        raw_scores = np.reshape(classificators, int(classificators.shape[0] * classificators.shape[1] * classificators.shape[2]))
        detections = ProcessCPU(raw_boxes, raw_scores, self.anchors, self.options)

        # NOTE: here we are only processing first detection for speed
        detections = orig_nms(detections, 0.3)
        for detection in detections:
            x1 = int(img_width * detection.xmin)
            x2 = int(img_width * (detection.xmin + detection.width))
            y1 = int(img_height * detection.ymin)
            y2 = int(img_height * (detection.ymin + detection.height))
            return [x1, y1, x2, y2]
