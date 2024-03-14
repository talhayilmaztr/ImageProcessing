import cv2


vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/talhayilmaz/Desktop/OpenCv/Opp/haarCascade/fire_cascade.xml")

while True:
    _,frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame,1)

    faces = face_cascade.detectMultiScale(gray,1.3,20)

    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("video", frame)
    if cv2.waitKey(10) & 0xff == ord("q"):
        break

vid.release
cv2.destroyAllWindows()


