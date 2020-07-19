# Face and Gesture Lock
## Aim
The project is aimed to provide a better lock system with different user experience. This lock system uses face recognition and unique air hand gesture as lock key.

## Overview
1. The user has to set a face and a unique gesture as unlock key.
    1. The user's face is saved in the form of feature vector.
    2. the gesture embeddings will be saved as returned by hand landmark recognizer.
2. Then for the unlock process:
    1. System initially monitors for faces ones it finds a face it sends croped face to the face recognizer.
    2. The face recognizer checks if the face is the one of the user if it finds it, it will start the gesture verification process.
    3. The gesture recognizer will monitor for the specified gesture it does not unlock the system untill it finds the gesture.

### todo list
- [ ] the modules directory should contain:
    - [ ] Face Detector.
    - [ ] Face Recognizer.
    - [X] Hand Landmark/Gesture Recognizer.
    - [ ] I/O module
- [ ] each module should have:
    - [ ] Preprocessing Function
    - [ ] predict method
    - [ ] Readme containing model summary, usage, etc.
    - [ ] Data folder containing the model pre-trained files.
    - [ ] Model Initializer Function
    - [ ] Model remove Function
- [ ] Design a single script for each models testing
- [ ] Single script for complete run
- [ ] Try using all this in Tensorflow.js