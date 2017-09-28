#-*-coding:utf-8-*-
#第三题

import cv2
import numpy as np
from drawtrack import drawMatches
from drawtrack import drawPoints
from drawtrack import drawAll
from drawtrack import drawMask
import copy
import math
import random

flag = 0

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 20,     # 20
                   multi_probe_level = 2) #2

MIN_MATCH_COUNT = 4
KNN_COUNT = 2

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

kernelopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)) #开运算的核：4x4方
kernelclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(2,2)) #闭运算的核：4x4方
detector = cv2.ORB() #ORB特征提取法
matcher = cv2.FlannBasedMatcher(flann_params, {}) #神奇的快速最近邻逼近搜索函数库(Fast Approximate Nearest Neighbor Search Library)匹配法

cap = [0,0,0,0]
cap[0]=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\5\\4p-c0.avi")
cap[1]=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\5\\4p-c1.avi")
cap[2]=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\5\\4p-c2.avi")
cap[3]=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\5\\4p-c3.avi")

fgbg = []
for i in range(4):
    fgbg0 = cv2.BackgroundSubtractorMOG()
    fgbg.append(fgbg0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) #开闭运算的核：3x3圆

first_colors = []
colors = []
count = 0

save_num = 1

while 1:
    count = count + 1
    
    success = []
    frame = []
    
    for i in range(4):
        success0,frame0=cap[i].read()
        success.append(success0)
        frame.append(frame0)
    
    if success[0]:
        for i in range(4):
            frame[i] = cv2.cvtColor(frame[i],cv2.COLOR_BGR2GRAY)
            
    else:
        break
    
    framethre = []
    for i in range(4):
        fgmask0 = fgbg[i].apply(frame[i])
        framethre0 = fgmask0
        framethre0 = cv2.morphologyEx(framethre0,cv2.MORPH_CLOSE,kernelclose) #闭运算
        con,hie = cv2.findContours(framethre0,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #找所有外轮廓
        cv2.drawContours(framethre0,con,-1,(255,255,255),-1) #填充所有外轮廓
        framethre0 = cv2.morphologyEx(framethre0,cv2.MORPH_OPEN,kernelopen) #开运算
        framethre.append(framethre0)

    frame_points = []
    frame_descrs = []
    
    for i in range(4):
        frame_points0, frame_descrs0 = detector.detectAndCompute(frame[i], None)
        frame_points.append(frame_points0)
        frame_descrs.append(frame_descrs0) 
    
    if len(frame_points[0]) < MIN_MATCH_COUNT:
        break
    if len(frame_points[1]) < MIN_MATCH_COUNT:
        break
    if len(frame_points[2]) < MIN_MATCH_COUNT:
        break
    if len(frame_points[3]) < MIN_MATCH_COUNT:
        break
    
    matches = []
    matches0 = matcher.knnMatch(frame_descrs[0],frame_descrs[1], k = 2) #特征点匹配,前query后train
    matches.append(matches0)
    matches0 = matcher.knnMatch(frame_descrs[1],frame_descrs[2], k = 2) #特征点匹配,前query后train
    matches.append(matches0)
    matches0 = matcher.knnMatch(frame_descrs[2],frame_descrs[3], k = 2) #特征点匹配,前query后train
    matches.append(matches0)
    matches0 = matcher.knnMatch(frame_descrs[3],frame_descrs[0], k = 2) #特征点匹配,前query后train
    matches.append(matches0)
    
    for i in range(4):
        matches[i] = [m[0] for m in matches[i] if len(m) == 2 and m[0].distance < m[1].distance * 0.7]

    if len(matches[0]) < MIN_MATCH_COUNT: #有效匹配的点对不够多，放弃ORB方法
        continue
    if len(matches[1]) < MIN_MATCH_COUNT: #有效匹配的点对不够多，放弃ORB方法
        continue
    if len(matches[2]) < MIN_MATCH_COUNT: #有效匹配的点对不够多，放弃ORB方法
        continue
    if len(matches[3]) < MIN_MATCH_COUNT: #有效匹配的点对不够多，放弃ORB方法
        continue
    
    c_matches = []
    M = []
    
    for i in range(4):
        
        if i == 0:
            q = 0
            t = 1
        if i == 1:
            q = 1
            t = 2
        if i == 2:
            q = 2
            t = 3
        if i == 3:
            q = 3
            t = 0
        
        src_pts = [frame_points[q][m.queryIdx].pt for m in matches[i]]
        dst_pts = [frame_points[t][m.trainIdx].pt for m in matches[i]]
        src_pts, dst_pts = np.float32((src_pts, dst_pts))
    
        M0, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        mask = np.squeeze(mask, 1)
         
        cc_matches = []
        for j in range(len(mask)):
            if mask[i] != 0:
                try:
                    cc_matches.append(matches[i][j])
                except:
                    print matches[i][j]
        
        c_matches.append(cc_matches)
        M.append(M0)

    thre_img = drawMask(framethre[0],framethre[1],framethre[2],framethre[3]) 
    frame_img = drawMask(frame[0],frame[1],frame[2],frame[3])
    mask_img = frame_img & thre_img 
    
    rail_img = drawAll(frame[0],frame_points[0],frame[1],frame_points[1],frame[2],frame_points[2],frame[3],frame_points[3],c_matches)    
    #rail_img = drawMatches(frame[0],frame_points[0],frame[1],frame_points[1],c_matches[0])
    #rail_img = drawPoints(frame[0],frame_points[0],frame[1],frame_points[1])
    
    
    #framethre = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel) #闭运算
    #con,hie = cv2.findContours(framethre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#找轮廓
    #cv2.drawContours(framethre,con,-1,(255,255,255),-1)#填轮廓
    #framethre = cv2.morphologyEx(framethre,cv2.MORPH_OPEN,kernel)#开运算
    #framenew = frame.copy()
    #framenew[:,:,0] = frame[:,:,0] & framethre
    #framenew[:,:,1] = frame[:,:,1] & framethre
    #framenew[:,:,2] = frame[:,:,2] & framethre
    
    cv2.imshow("out", rail_img)
    
    #cv2.imshow("input", frame_img)
    #cv2.imshow("output", mask_img)
    #cv2.imshow("outputbin", thre_img)
        
        
    if cv2.waitKey(0) & 0xFF == ord(' '):
        print count
        print M[0]
        print M[1]
        print M[2]
        print M[3]
        #cv2.imwrite("input" + str(save_num) + ".jpg", frame_img)
        #cv2.imwrite("output" + str(save_num) + ".jpg", mask_img)
        #cv2.imwrite("outputbin" + str(save_num) + ".jpg", thre_img)
        #save_num = save_num + 1

print count
cap[0].release()
cap[1].release()
cap[2].release()
cap[3].release()
cv2.destroyAllWindows()