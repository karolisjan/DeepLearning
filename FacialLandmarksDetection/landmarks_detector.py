# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 00:35:05 2017

@author: Karolis
"""
import cv2
import numpy as np
from keras.models import load_model


class LandmarksDetector():
    def __init__(self, cnn_model_path, face_cascade_path):
        
        self.__cnn_model = load_model(cnn_model_path)
        self.__face_detector = cv2.CascadeClassifier(face_cascade_path)

    def detect(self, 
        img, 
        scale_factor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    ):
        '''
            scaleFactor  - parameter specifying how much the image size is reduced
                           at each image scale, i.e. is used to creat a scale 
                           pyramid.
                          
            minNeighbors - how many neighbors each candidate rectangle should 
                           have to retain.
                           
            minSize      - minimum possible object size. Objects smaller than
                           minSize are ignored.
        '''            
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        rectangles = self.__face_detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        
        all_landmarks = []
        
        for x, y, w, h in rectangles:
            x_min, x_max = x, x + w
            y_min, y_max = y, y + h
            
            roi = gray[y_min:y_max, x_min:x_max]
            roi = cv2.resize(roi, (96, 96)) 
            roi = roi.astype(float) / 255.0
            roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)
            
            landmarks = self.__cnn_model.predict(roi)
            landmarks = np.squeeze(landmarks)
            
            # x coordinates
            landmarks[0::2] = (landmarks[0::2] * 48 + 48) * (x_max - x_min) / 96 + x_min
            
            # y coordinates
            landmarks[1::2] = (landmarks[1::2] * 48 + 48) * (y_max - y_min) / 96 + y_min
        
            all_landmarks.append(landmarks)
            
        return rectangles, all_landmarks