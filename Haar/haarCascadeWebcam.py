import cv2
from functions import get_faces_with_eyes

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _,frame=cap.read()
    get_faces_with_eyes(frame)
    cv2.imshow("window",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

