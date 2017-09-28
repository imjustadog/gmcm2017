#-*-coding:utf-8-*-

#第二题

import cv2
import numpy as np
from draw import draw_line
flag = 0

#cap=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\dynamic-without-swing\\waterSurface\\input.avi")
cap=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\dynamic-without-swing\\waterSurface\\input.avi")
fgbg = cv2.BackgroundSubtractorMOG() #混合高斯模型
kernelopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)) #开运算的核：4x4方
kernelclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(2,2)) #闭运算的核：4x4方
success,frame=cap.read()
save_num = 1

hist = []
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('output.avi',-1,fps,size)

while success:
    success,frame=cap.read()
    if success:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
    if flag == 0:
        flag = 1
        fgmask = fgbg.apply(frame)
        framepro = frame
    
    else:
        fgmask = fgbg.apply(frame)
        framethre = fgmask
        framethre = cv2.morphologyEx(framethre,cv2.MORPH_CLOSE,kernelclose) #闭运算
        con,hie = cv2.findContours(framethre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #找所有外轮廓
        cv2.drawContours(framethre,con,-1,(255,255,255),-1) #填充所有外轮廓
        framethre = cv2.morphologyEx(framethre,cv2.MORPH_OPEN,kernelopen) #开运算
        
        frame_mask = frame.copy()
        frame_mask = frame & framethre
        
        white = (framethre != 0)
        white = sum(map(sum,white))
        hist.append(white)
        
        cv2.imshow("input", frame)
        cv2.imshow("output", frame_mask)
        cv2.imshow("outputbin", framethre)
        out.write(frame_mask)
        
    if cv2.waitKey(1) & 0xFF == ord(' '):
        #cv2.imwrite("exm_input" + str(save_num) + ".jpg", frame)
        #cv2.imwrite("exm_output" + str(save_num) + ".jpg", framenew)
        #cv2.imwrite("exm_outputbin" + str(save_num) + ".jpg", framethre)
        #save_num = save_num + 1
        break

cap.release() 
out.release()    
cv2.destroyAllWindows()

#ret = draw_line(hist)
#file = open('data_campus.txt','w')
#file.write(str(hist))
#file.close()
