import cv2

"""
img = cv2.imread("1.jpeg",0)
print(img)
#cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resize(img, (640,480))

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("2.jpeg", img)
"""
cap = cv2.VideoCapture(0)

filename = "/Users/talhayilmaz/Desktop/OpenCv/opp/merhaba.mp4"
codec = cv2.VideoWriter_fourcc(*"mp4v")
frameRate = 30
resolution = (640 ,480)

videoFileOutput = cv2.VideoWriter(filename, codec, frameRate, resolution)
while True:
    ret, frame = cap.read()
    if ret == 0:
        break
    frame = cv2.flip(frame,1)
    videoFileOutput.write(frame)
    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

videoFileOutput.release()
#cap.release()
cv2.destroyAllWindows

