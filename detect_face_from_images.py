# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:13:56 2019

@author: bzcvwz
"""

import cv2
import numpy as np
import argparse

av = argparse.ArgumentParser()
av.add_argument('-i','--image',help='enter imagefile path',required=True)
av.add_argument('-p','--proto',help='enter protofile',required=True)
av.add_argument('-m','--model',help='enter model file',required=True)
av.add_argument('-c','--confidence_threshold',type=float,help='enter confidence threshold',default=0.5)

#args=av.parse_args()
#print(args)
#Namespace(confidence_threshold=0.5, image='rooster.jpg', model='res10_300x300_ssd_iter_140000.caffemodel', proto='deploy.prototxt.txt')

args=vars(av.parse_args())
#returns the dict attribute
#print(args)
#{'image': 'rooster.jpg', 'proto': 'deploy.prototxt.txt', 'model': 'res10_300x300_ssd_iter_140000.caffemodel', 'confidence_threshold': 0.5}

#print(args['proto'])
#print(args['model'])
#print(args['confidence_threshold'])

img = cv2.imread(args['image'])
#cv2.imshow('image',img)
#cv2.waitKey(0)
#print(img.shape)
(h,w) =img.shape[0:2]
#print(h)
#print(w)
blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

print('[Info]Loading model...')
net=cv2.dnn.readNetFromCaffe(args['proto'],args['model'])

net.setInput(blob)
detections = net.forward()

for i in range(0,detections.shape[2]):
    
    prob = detections[0,0,i,2]
    
    if prob >= args['confidence_threshold']:
        #print('found a face')
        bbox = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY)=bbox.astype(int)
        #print(startX,startY,endX,endY)
        
        cv2.rectangle(img,(startX,startY),(endX,endY),(0, 0, 255),2)
        
        txt = '{:.2f}%'.format(prob*100)
        y_txt = startY-10 if startY-10 > 10 else startY+10
        #text = "{:.2f}%".format(confidence * 100)
        cv2.putText(img,txt,(startX,y_txt),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.45,color=(255,0,0),thickness=2)
        
        
cv2.imshow('image',img)
cv2.waitKey(0)
