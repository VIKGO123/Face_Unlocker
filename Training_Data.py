#create training data
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier("C:/Users/hp/Desktop/AI/haarcascade_frontalface_default.xml")

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    print(faces)
    
    if(faces==()):
        return faces
    #crop all faces found
    for(x,y,w,h) in faces:
        img=img[y:y+h,x:x+w]
        return img

cap=cv2.VideoCapture(0)
count=0

#collect 100 samples of your face from webcam
while(True):
    ret,frame=cap.read()

    if ret is True:
        if(face_extractor(frame)==()):
            print("face not found")
        
        else:
            count=count+1
            faces=cv2.resize(face_extractor(frame),(200,200))
            face=cv2.cvtColor(faces,cv2.COLOR_BGR2GRAY)
        
        #SAVE FILE IN SPECIFIED LOC WITH UNIQUE NAME
            file_name_path=str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)
        
        
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("face",face)

        if cv2.waitKey(1)==13 or count==100:#repeat this loop to create any sets of images
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()
print("complete")    
    
    
    
