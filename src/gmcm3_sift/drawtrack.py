import numpy as np
import cv2
import random
import math

last_match = []
last_color = []

all_track = {}

first_flag = 1

def drawtracks(img,rail, kp1, kp2, matches,colors):
    global last_match
    global last_color
    global first_flag
    global all_track
    global num_track
    
    if first_flag == 1:
        last_match = []
        last_color = []
        first_flag = 0
        for mat in matches:
            last_match.append(mat.trainIdx)
    
    else:
        current_match = []
        for mat in matches:
            last_index = -1
            try:
                last_index = last_match.find(mat.queryIdx)
                current_match.append(mat.trainIdx)
            except:
                pass
            
            
            # x - columns
            # y - rows
            (x1,y1) = kp1[mat.queryIdx].pt
            (x2,y2) = kp2[mat.trainIdx].pt
            i1 = int(colors[mat.trainIdx][0])
            i2 = int(colors[mat.trainIdx][1])
            i3 = int(colors[mat.trainIdx][2])
            
            
            if i1 == 255 and i2 == 255 and i3 == 255:
                pass
            else:
                cv2.circle(img, (int(x2),int(y2)), 4, (i1, i2, i3), 1) 
                cv2.circle(rail, (int(x2),int(y2)), 2, (i1, i2, i3), -1)
                cv2.line(rail, (int(x1),int(y1)), (int(x2),int(y2)), (i1, i2, i3), 1)    
            
        last_match = current_match