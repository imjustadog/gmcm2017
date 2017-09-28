#-*-coding:utf-8-*-

#第四题

import cv2
import numpy as np

cap=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\examine\\Escalator\\escalator.avi")
success,frame=cap.read()
count = 0
wait_time = 1

while success:
    success,frame=cap.read()
    if success:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        count = count + 1
    else:
        break
    
    if count < 2781:
        continue
    elif count > 3415:
        wait_time = 100
        cv2.imshow("input", frame)
        print 'finish'
    else:
        wait_time = 1
        cv2.imshow("input", frame)
        
    if cv2.waitKey(wait_time) & 0xFF == ord(' '):
        #cv2.imwrite("exm_input" + str(save_num) + ".jpg", frame)
        #cv2.imwrite("exm_output" + str(save_num) + ".jpg", framenew)
        #cv2.imwrite("exm_outputbin" + str(save_num) + ".jpg", framethre)
        #save_num = save_num + 1
        break

cap.release()    
cv2.destroyAllWindows()
