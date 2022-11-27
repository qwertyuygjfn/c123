import cv2
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl
import time
x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
xtrainscaled=xtrain/255
xtestscale=xtest/255
xmodel=LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscaled,ytrain)
ypred=xmodel.predict(xtestscale)
accuracy=accuracy_score(ytest,ypred)
print("accuracy is ",accuracy)
cap=cv2.VideoCapture(0)
print("camera started")
while (True):
        
    try:
        ret,frame=cap.read()
        print("Inside the loop")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(widht/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi=gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        impil=Image.fromarray(roi)
        imagebw=impil.convert("L")
        imagebwresize=imagebw.resize((28,28),Image.ANTIALIAS)
        imagebwresizeinverted=PIL.ImageOps.invert(imagebwresize)
        pixel_filter=20
        minpix=np.percentile(imagebwresizeinverted,filterpixel)
        imagebwresizeinvertedscaled=np.clip(imagebwresizeinverted-minpix,0,255)
        maxpix=np.max(imagebwresizeinverted)
        imagebwresizeinvertedscaled=np.asarray(imagebwresizeinvertedscaled)/maxpix
        testsample=np.array(imagebwresizeinvertedscaled).reshape(1,784)
        testpred=xmodel.predict(testsample)
        print("predicted class is: ",testpred)
        cv2.imshow("frame",gray)
        if cv2.wait(1)&0xFF==ord("q"):
            break
    except Exception  as e:
        pass  
cap.release()
cv2.destroyAllWindows()    
        
