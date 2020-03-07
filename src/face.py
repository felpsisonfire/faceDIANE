import cv2 
import numpy as np

cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt2.xml')

if __name__ == "__main__":

    video = cv2.VideoCapture('footage.mp4')

    while(video.isOpened()):
        ret, frame = video.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.5, 5)
            for (x, y, w, h) in faces:  
                roi = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

video.release()
cv2.destroyAllWindows()
