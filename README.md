# Face and Gesture Lock
The project is aimed to provide a better lock system with different user experience. This lock system uses face recognition and unique air hand gesture as lock key.


## Aim
This project is an attempt to provide a unique locking system to the users while also increasing the security.

Ever thought what if someone could show your image or suddenly point the mobile phone to your face just to peep into your private life?... This project is designed just to ensure that never happens to anyone else.


![](Demo.gif)



## Overview
1. The user has to set a face and a unique gesture as unlock key.
    1. The user's face is saved in the form of feature vector.
    2. The gesture embeddings will be saved as returned by hand landmark recognizer.
2. Then for the unlock process:
    1. System initially monitors for faces ones it finds a face it sends croped face to the face recognizer.
    2. The face recognizer checks if the face is the one saved by the user if it is the face is marked as Matched.
    3. The gesture recognizer will monitor for the specified gesture if it finds it then the gesture is Marked as match.
    4. When Both the Face and The Gesture are matched the system is unlocked.

## Usage
1. Clone the repository at a local directory.
    ```
    $ git clone https://github.com/Mr-MayankThakur/Face-and-Gesture-Lock.git
    ```
2. Install Requirements
    ```
    $ pip install -r requirements.txt
    ```
3. Save 
    - Face Image in data/face_data/face_images
    - Hand Gesture Image in data/geature_data/gesture_images
    
4. Generate Face and Gesture Embeddings which will be compared while unlocking
    ```commandline
    $ python save_face_embeddings.py
    $ python save_hand_gestures.py
    ```

5. Run the Recognizer
    ```
   $ python run_face_and_gesture_recognizer.py
    ```


## Advantages

- Advanced and Key Free Security System.
- Multiple People can be recognized by the system.
- We could use different gestures for different tasks while also allowing authorized personal only.
- Each Person can have Unique Gesture.  
- There are many more use cases your imagination is the limit.


## Limitations

- Only one hand is monitored at a time for Hand Gesture.

## Special Thanks
- **[Laurence Moroney](https://www.linkedin.com/in/laurence-moroney/), AI Lead at Google, all this was not possible without his Motivation and Teaching. Respect** :bow:

- **[Suyash Tople](https://www.linkedin.com/in/suyash1999/)** for Testing and Collaboration.
