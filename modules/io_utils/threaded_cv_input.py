# source: https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56

from threading import Thread, Lock
import cv2 as cv

class WebcamVideoStream :
    def __init__(self, src = 0, width = 640, height = 480) :
        self.stream = cv.VideoCapture(src)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def show(self, win_name, img):
        cv.imshow(win_name, img)

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    while True :
        frame = vs.read()
        #cv.imshow('webcam', frame)
        vs.show('webcam', frame)
        if cv.waitKey(1) == 27 :
            break

    vs.stop()
    cv.destroyAllWindows()