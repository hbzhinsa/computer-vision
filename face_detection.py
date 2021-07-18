#%%
import cv2 as cv
# %%
img=cv.imread('people.jpeg')
cv.imshow('person',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray person', gray)

# %%
haar_cascade=cv.CascadeClassifier('haar_face.xml')
face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=2)
print(f'Number of faces found={len(face_rect)}')
for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0), thickness=2)
cv.imshow(' Detectd faces',img)
cv.waitKey(0)
# haarcascade is very sensitive to noise..