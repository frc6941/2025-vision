import time
import cv2

camera_id: int = int(input("input camera id (should be int)\n"))
cap = cv2.VideoCapture(camera_id)

while True:
    ret, image = cap.read()
    if(ret):
        cv2.imshow("Image", image)
        cv2.waitKey(1)
    else:
        print("Ret False, Check id")
        time.sleep(1000)
