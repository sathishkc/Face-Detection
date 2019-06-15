# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:21:42 2019

@author: bzcvwz
"""

from imutils.video import FileVideoStream
#from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-v','--video',required=True,help='path to video file')
ap.add_argument('-m','--model',required=True,help='path to model')
ap.add_argument('-p','--proto',required=True,help='path to proto file')
ap.add_argument('-c','--confidence',default=0.4,type=float)

args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args['proto'],args['model'])

fvs = FileVideoStream(args['video']).start()
time.sleep(2)

#fps = FPS().start()

while fvs.more():
    
    frame = fvs.read()
    
    frame = imutils.resize(frame,width=400)
    
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    #print(detections.shape)
    for i in range(0, detections.shape[2]):
    
        conf = detections[0,0,i,2]
        
        if(conf > args['confidence']):
            
            bbox = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = bbox.astype("int")
            
            cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),3)
            
            txtY= startY-10 if startY-10 > 10 else startY+10
            
            txt = "{:.2f}%".format(conf*100)
            txt = cv2.putText(frame,txt,(startX,txtY),cv2.FONT_HERSHEY_COMPLEX,0.45, (0, 0, 255), 2)
            
    cv2.imshow('stream',frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
fvs.stop()
