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

cap=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\6.avi")
fgbg = cv2.BackgroundSubtractorMOG() #混合高斯模型
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) #开闭运算的核：3x3圆
success,frame=cap.read()
rail_img = np.full((frame.shape[0],frame.shape[1],3),255, dtype='uint8')
limit_high = math.sqrt(rail_img.shape[0] * rail_img.shape[0] + rail_img.shape[1] * rail_img.shape[1])/10

first_colors = []
colors = []
count = 0

while success:
    count = count + 1
    success,frame=cap.read()
    if success:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        cv2.imwrite("output7.jpg", rail_img)
    
    if flag == 0:
        flag = 1
        frame_points, frame_descrs = detector.detectAndCompute(frame, None) #找出当前画面的所有特征点
        frame_points_first = frame_points
        frame_descrs_first = frame_descrs
        frame_points_pre = frame_points
        frame_descrs_pre = frame_descrs
        for i in range(len(frame_points_first)):
            i1 = random.randint(0, 254)
            i2 = random.randint(0, 254)
            i3 = random.randint(0, 254) 
            first_colors.append([i1,i2,i3])
        
    else:
        frame_points, frame_descrs = detector.detectAndCompute(frame, None) #找出当前画面的所有特征点
        if len(frame_points) < MIN_MATCH_COUNT: #找到的特征点不够多，放弃ORB方法
            continue
        
        matches = matcher.knnMatch(frame_descrs_pre,frame_descrs, k = KNN_COUNT) #特征点匹配
        matches_color = matcher.knnMatch(frame_descrs_first,frame_descrs, k = 1) #特征点匹配
        
        if len(matches_color) < MIN_MATCH_COUNT:
            continue
        
        
        good = []
        for item in matches:
            if len(item) < KNN_COUNT:
                continue
            A = [frame_points_pre[m.queryIdx].pt for m in item]
            B = [frame_points[m.trainIdx].pt for m in item] 
            A = np.array(map(list,A))
            B = np.array(map(list,B))
            C = A - B
            C = C ** 2
            C = C.sum(axis = 1)
            C = C ** 0.5
            for index in range(len(item) - 1):
                if C[index] > limit_high:
                    continue
                if item[index].distance < 50.0:
                    good.append(item[index])
                break
        
        if len(good) < MIN_MATCH_COUNT:
            continue
                        
        total = len(frame_points_first)
        
        src_pts_color = []
        dst_pts_color = []
        for m in matches_color:
            if m == []:
                continue
            if m[0].queryIdx < total:
                src_pts_color.append(frame_points_first[m[0].queryIdx].pt)
                dst_pts_color.append(frame_points[m[0].trainIdx].pt)
        
        src_pts_color = np.float32(src_pts_color).reshape(-1,1,2)
        dst_pts_color = np.float32(dst_pts_color).reshape(-1,1,2)
        
        M, mask_color = cv2.findHomography(src_pts_color, dst_pts_color, cv2.RANSAC,3.0)
        mask_color = np.squeeze(mask_color, 1)
        
        cc_matches_color = []
        for i in range(len(mask_color)):
            if mask_color[i] != 0:
                try:
                    cc_matches_color.append(matches_color[i][0])
                except:
                    print matches_color[i]
    
        colors = np.full((len(frame_points),3),255, dtype='uint8')
        for item in cc_matches_color:
            if item.queryIdx > total:
                continue
            elif item.distance < 50.0:
                colors[item.trainIdx] = first_colors[item.queryIdx]
            
        
        src_pts = np.float32([frame_points_pre[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([frame_points[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
        mask = np.squeeze(mask, 1)
        
        cc_matches = []
        for i in range(len(mask)):
            if mask[i] != 0 :
                cc_matches.append(good[i])
        
        drawtracks(frame,rail_img, frame_points_pre, frame_points,cc_matches,colors)
        #framethre = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel) #闭运算
        #con,hie = cv2.findContours(framethre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#找轮廓
        #cv2.drawContours(framethre,con,-1,(255,255,255),-1)#填轮廓
        #framethre = cv2.morphologyEx(framethre,cv2.MORPH_OPEN,kernel)#开运算
        #framenew = frame.copy()
        #framenew[:,:,0] = frame[:,:,0] & framethre
        #framenew[:,:,1] = frame[:,:,1] & framethre
        #framenew[:,:,2] = frame[:,:,2] & framethre
        cv2.imshow("input", frame)
        cv2.imshow("output", rail_img)
        
        frame_points_pre = frame_points
        frame_descrs_pre = frame_descrs
        
        
    if cv2.waitKey(10) & 0xFF == ord(' '):
        cv2.imwrite("orb.jpg", frame)
        break

cv2.imwrite("track.jpg", rail_img)
cap.release()    
cv2.destroyAllWindows()