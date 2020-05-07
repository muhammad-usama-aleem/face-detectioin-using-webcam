import os
import cv2


os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")
video = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    check, Frame = video.read()
    gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    for x, y, w, h in img:
        rec = cv2.rectangle(Frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    final = cv2.resize(Frame, (int(Frame.shape[1] / 1.5), int(Frame.shape[0] / 1.5)))
    show = cv2.imshow("live", final)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
