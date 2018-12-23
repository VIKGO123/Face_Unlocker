import numpy as np
import cv2
from os import listdir
from os.path import join,isfile

#get training data we previously made
data_path="C:/Users/hp/Desktop/AI/user/"    #path in which you saved your training data
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

#arrays for taining data and labels
Training_data,Labels=[],[]

for i ,files in enumerate(onlyfiles):
    image_path=data_path+files
    images=cv2.imread(image_path,0)
   # cv2.imshow("image",images)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels=np.asarray(Labels,dtype=np.int32)

#initialise facial recogniser
model=cv2.createLBPHFaceRecognizer()

#train_model
model.train(np.asarray(Training_data),np.asarray(Labels))
print("model trained successfully")

face_classifier=cv2.CascadeClassifier("C:/Users/hp/Desktop/AI/Udemy CV/haarcascade_frontalface_default.xml")

def face_detector(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)


    if(faces==()):
        return img,[]
    #crop all faces found
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
        return img,roi
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if(ret==True):
        image,face=face_detector(frame)

        try:
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        #RESULT COMPRISES OF A TUPLE CONTAINING LABEL ANF CONFIDENCE VALUE
            results=model.predict(face)
           
            if(int(results[1])<500):
                confidence=int(100*(1-(results[1]/300)))#probability of recognition
                print(confidence)
                display_str="confidence about user"+str(confidence)
                cv2.putText(image,display_str,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(120,255,255),2)

                if(confidence>70):
                    cv2.putText(image,"unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.imshow("face cropper",image)
                    
                else:
                    cv2.putText(image,"locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.imshow("face cropper",image)
        except:
            cv2.putText(image,"no face found",(220,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.putText(image,"locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("face cropper",image)
            pass

        if cv2.waitKey(1)==13:
            break
    else:
        continue

cap.release()
cv2.destroyAllWindows()
