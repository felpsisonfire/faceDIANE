import cv2
import numpy as np
import sys

def process_frame(img, cascade, show = True):
    """ Process the frames and show the prediction"""
    while(img.isOpened()):
        ret, frame = img.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.5, 5)
            for(x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if show == True:
                cv2.imshow('frame', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

if __name__ == "__main__":
    try:
        video = cv2.VideoCapture(sys.argv[1])
    except IndexError:
        print("Error: Video file not pass\n Try:") 
        print("python3 diane.py [videofile] [cascadeClassifier]")
    try:
        cascade = cv2.CascadeClassifier(sys.argv[2])
    except IndexError:
        print("Error: Video file not pass\n Try:") 
        print("python3 diane.py [videofile] [cascadeClassifier]")

    process_frame(video, cascade)
