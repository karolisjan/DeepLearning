# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 00:56:10 2017

@author: Karolis
"""
import cv2
import time
import imageio
import imutils
import argparse
from landmarks_detector import LandmarksDetector

        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-c',
        '--cascade_path',
        required=True,
        help='path to the face haar cascade'
    )

    ap.add_argument(
        '-m',
        '--model_path',
        required=True,
        help='path to a trained CNN smile model'
    )

    ap.add_argument(
        '-v',
        '--video_path',
        help='path to a video file (optional)'
    )

    ap.add_argument(
        '-o',
        '--output_path',
        help='path to save the output to (optional)'
    )
       
    
    args = ap.parse_args()
       
    model = LandmarksDetector(args.model_path, args.cascade_path)
    camera = cv2.VideoCapture(0) if 'video_path' not in args else cv2.VideoCapture(args.video_path)
    output = []
            
    while True:
        _, frame = camera.read()

        if frame is None:
            break
       
        frame_copy = frame.copy()
        rectangles, all_landmarks = model.detect(frame_copy)
        
        for (x, y, w, h), landmarks in zip(rectangles, all_landmarks):
            # Draw a rectangle around the face
            cv2.rectangle(
                frame_copy, 
                (x, y), 
                (x + w, y + h), 
                (0, 0, 255), 
                2
            )
            
            # Mark the landmakrs
            for (xx, yy) in zip(landmarks[0::2], landmarks[1::2]):
                cv2.circle(
                    frame_copy, 
                    (xx, yy),
                    3,
                    (0, 0, 255), 
                    -1
                )
        
        cv2.imshow('Frame', frame_copy)
        output.append(frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    if 'output_path' in args:
        with imageio.get_writer(args.output_path, mode='I') as writer:
            for frame in output:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame)

    camera.release()
    cv2.destroyAllWindows()
