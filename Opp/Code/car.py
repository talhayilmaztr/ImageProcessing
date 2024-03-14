import cv2


vid = cv2.VideoCapture("/Users/talhayilmaz/Desktop/OpenCv/Opp/test_videos/car.mp4")
face_cascade = cv2.CascadeClassifier("/Users/talhayilmaz/Desktop/OpenCv/Opp/haarCascade/cars.xml")

while True:
    _,frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)

    cv2.imshow("video", frame)
    if cv2.waitKey(10) & 0xff == ord("q"):
        break

vid.release
cv2.destroyAllWindows()


