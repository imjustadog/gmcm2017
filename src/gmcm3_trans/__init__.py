#-*-coding:utf-8-*-
#第三题

import cv2
import numpy as np
from drawtrack import drawtracks
import copy
import math
import random

flag = 0

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10
KNN_COUNT = 10

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)


detector = cv2.ORB() #ORB特征提取法
matcher = cv2.FlannBasedMatcher(flann_params, {}) #神奇的快速最近邻逼近搜索函数库(Fast Approximate Nearest Neighbor Search Library)匹配法

cap=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\with-swing\\cars7\\input.avi")
fgbg = cv2.BackgroundSubtractorMOG() #混合高斯模型
success,frame=cap.read()
out_img = np.full((frame.shape[0],frame.shape[1],3),255, dtype='uint8')
kernelopen = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #开运算的核：4x4方
kernelclose = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #闭运算的核：4x4方
fgbg = cv2.BackgroundSubtractorMOG() #混合高斯模型
save_num = 1

fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
#out = cv2.VideoWriter('output.avi',-1,fps,size)

while success:
    success,frame=cap.read()
    if success:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
        
    
    if flag == 0:
        flag = 1
        frame_points, frame_descrs = detector.detectAndCompute(frame, None) #找出当前画面的所有特征点
        frame_points_first = frame_points
        frame_descrs_first = frame_descrs
        
    
    elif flag == 1:
        flag = 2
        frame_points, frame_descrs = detector.detectAndCompute(frame, None) #找出当前画面的所有特征点
        if len(frame_points) < MIN_MATCH_COUNT: #找到的特征点不够多，放弃ORB方法
            continue
        matches = matcher.knnMatch(frame_descrs_first,frame_descrs, k = 2) #特征点匹配
        good = []
        for item in matches:
            if len(item) < 2:
                break
            m = item[0]
            n = item[1]
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good) < MIN_MATCH_COUNT:
            continue
        
        src_pts = np.float32([frame_points_first[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([frame_points[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M = cv2.estimateRigidTransform(dst_pts, src_pts,False)
        
        out_img_pre = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        fgmask = fgbg.apply(out_img_pre)
        
        
    else:
        frame_points, frame_descrs = detector.detectAndCompute(frame, None) #找出当前画面的所有特征点
        
        if len(frame_points) < MIN_MATCH_COUNT: #找到的特征点不够多，放弃ORB方法
            continue
        matches = matcher.knnMatch(frame_descrs_first,frame_descrs, k = 2) #特征点匹配
        good = []
        for item in matches:
            if len(item) < 2:
                break
            m = item[0]
            n = item[1]
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good) < MIN_MATCH_COUNT:
            continue
        
        src_pts = np.float32([frame_points_first[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([frame_points[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M = cv2.estimateRigidTransform(dst_pts, src_pts,True)
        MT = cv2.estimateRigidTransform(src_pts, dst_pts,True)
        
        out_img = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        fgmask = fgbg.apply(out_img)
        framethre = fgmask
        framethre = cv2.morphologyEx(framethre,cv2.MORPH_CLOSE,kernelclose) #闭运算
        con,hie = cv2.findContours(framethre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #找所有外轮廓
        cv2.drawContours(framethre,con,-1,(255,255,255),-1) #填充所有外轮廓
        framethre = cv2.morphologyEx(framethre,cv2.MORPH_OPEN,kernelopen) #开运算
        
        framediff = cv2.absdiff(out_img,out_img_pre)
        ret, framethre_d = cv2.threshold(framediff,15,255,cv2.THRESH_BINARY) #二值化
        con,hie = cv2.findContours(framethre_d,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #找所有外轮廓
        cv2.drawContours(framethre_d,con,-1,(255,255,255),-1) #填充所有外轮廓
        framethre_d = cv2.morphologyEx(framethre_d,cv2.MORPH_OPEN,kernelopen) #开运算

        framethre_f = framethre & framethre_d
        #framethre_f = framethre_d

        framethre_f = cv2.warpAffine(framethre_f, MT, (frame.shape[1], frame.shape[0]))
        frame_mask = frame.copy()
        frame_mask = frame & framethre_f
        
        cv2.imshow("input", frame)
        cv2.imshow("output", frame_mask)
        cv2.imshow("outputbin", framethre_f)
        #out.write(frame_mask)
        
        out_img_pre = out_img.copy()
        
        if cv2.waitKey(100) & 0xFF == ord(' '):
            cv2.imwrite("input" + str(save_num) + ".jpg", frame)
            cv2.imwrite("output" + str(save_num) + ".jpg", frame_mask)
            cv2.imwrite("outputbin" + str(save_num) + ".jpg", framethre_f)
            save_num = save_num + 1


cap.release()  
out.release()   
cv2.destroyAllWindows()