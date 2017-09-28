import numpy as np
import os
import cv2

#vibe

flag = 0
N = 40
R = 40
_min = 5
phai = 16

def initial_background(I_gray, N):
    
    I_pad = np.pad(I_gray, 1, 'symmetric')
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    samples = np.zeros((height,width,N))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for n in range(N):
                x, y = 0, 0
                while(x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj]
    samples = samples[1:height-1, 1:width-1]
    return samples
    
def vibe_detection(I_gray, samples, _min, N, R):
    
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    segMap = np.zeros((height, width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            count, index, dist = 0, 0, 0
            while count < _min and index < N:
                dist = np.abs(I_gray[i,j] - samples[i,j,index])
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                r = np.random.randint(0, N-1)
                if r == 0:
                    r = np.random.randint(0, N-1)
                    samples[i,j,r] = I_gray[i,j]
                r = np.random.randint(0, N-1)
                if r == 0:
                    x, y = 0, 0
                    while(x == 0 and y == 0):
                        x = np.random.randint(-1, 1)
                        y = np.random.randint(-1, 1)
                    r = np.random.randint(0, N-1)
                    ri = i + x
                    rj = j + y
                    try:
                        samples[ri, rj, r] = I_gray[i, j]
                    except:
                        pass
            else:
                segMap[i, j] = 255
    return segMap, samples
    
cap=cv2.VideoCapture("C:\\Users\\zwq\\Desktop\\with-swing\\cars6\\input.avi")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
success,frame=cap.read()

while success:
    success,frame=cap.read()
    if flag == 0:
        flag = 1
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        samples = initial_background(gray, N)
    else:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        segMap, samples = vibe_detection(gray, samples, _min, N, R)
        framepro = frame
        cv2.imshow("input", segMap)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()    
cv2.destroyAllWindows()
