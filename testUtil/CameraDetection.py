import cv2

cv2.setLogLevel(0)
for i in range(10000):
    cap = cv2.VideoCapture(i)
    ret, image = cap.read()
    if ret:
        print(i)
