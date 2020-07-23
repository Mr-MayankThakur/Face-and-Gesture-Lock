# Face and Gesture Lock
## Aim
The project is aimed to provide a better lock system with different user experience. This lock system uses face recognition and unique air hand gesture as lock key.

## Overview
1. The user has to set a face and a unique gesture as unlock key.
    1. The user's face is saved in the form of feature vector.
    2. The gesture embeddings will be saved as returned by hand landmark recognizer.
2. Then for the unlock process:
    1. System initially monitors for faces ones it finds a face it sends croped face to the face recognizer.
    2. The face recognizer checks if the face is the one saved by the user if it is the face is marked as Matched.
    3. The gesture recognizer will monitor for the specified gesture if it finds it then the gesture is Marked as match.
    4. When Both the Face and The Gesture are matched the system is unlocked.