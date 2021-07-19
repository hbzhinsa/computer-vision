#%%
import os
import cv2 as cv
import numpy as np

people=['abe','bidon','obama','trump']
p=[]
dir=r'./faces'
for f in os.listdir(r'./faces'):
    if not f.startswith('.'):
        p.append(f)
print(p)
#%%
haar_cascade=cv.CascadeClassifier('haar_face.xml')
features=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)
            
            for (x,y,w,h) in face_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print(f'leanght of the featues={len(features)}')
print(f'leanght of the labels={len(labels)}')           
features=np.array(features,dtype='object')   
labels=np.array(labels)         
#%%           
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels) 
face_recognizer.save('face_recognizer.yml')
# %%
#* Try it
img=cv.imread('./faces_test/images-0.jpeg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)
faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    face_roi=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(face_roi)
    print(f'Label={label} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    cv.imshow('Detectd face',img)
cv.waitKey(0)
    
 

    


# %%
