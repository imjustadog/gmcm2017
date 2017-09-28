import numpy as np
import cv2
import random
import math

last_match = []
last_color = []

all_track = {}

first_flag = 1

def drawMask(img1, img2, img3, img4):
  
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((rows1+rows2,cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[rows1:,:cols1] = np.dstack([img4, img4, img4])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    out[rows2:,cols1:] = np.dstack([img3, img3, img3])

    return out


def drawAll(img1, kp1, img2, kp2, img3, kp3, img4, kp4, matches):
  
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((rows1+rows2,cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[rows1:,:cols1] = np.dstack([img4, img4, img4])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    out[rows2:,cols1:] = np.dstack([img3, img3, img3])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for index,match in enumerate(matches):
        for mat in match:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
    

            i1 = random.randint(0, 255)
            i2 = random.randint(0, 255)
            i3 = random.randint(0, 255)
            
            if index == 0:
                (x1,y1) = kp1[img1_idx].pt
                (x2,y2) = kp2[img2_idx].pt
                cv2.circle(out, (int(x1),int(y1)), 4, (i1, i2, i3), 1)   
                cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (i1, i2, i3), 1)
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (i1, i2, i3), 1)

            elif index == 1:
                (x1,y1) = kp2[img1_idx].pt
                (x2,y2) = kp3[img2_idx].pt
                cv2.circle(out, (int(x1)+cols1,int(y1)), 4, (i1, i2, i3), 1)   
                cv2.circle(out, (int(x2)+cols1,int(y2)+rows1), 4, (i1, i2, i3), 1)
                cv2.line(out, (int(x1)+cols1,int(y1)), (int(x2)+cols1,int(y2)+rows1), (i1, i2, i3), 1)

            elif index == 2:
                (x1,y1) = kp3[img1_idx].pt
                (x2,y2) = kp4[img2_idx].pt
                cv2.circle(out, (int(x1)+cols1,int(y1)+rows1), 4, (i1, i2, i3), 1)   
                cv2.circle(out, (int(x2),int(y2)+rows1), 4, (i1, i2, i3), 1)
                cv2.line(out, (int(x1)+cols1,int(y1)+rows1), (int(x2),int(y2)+rows1), (i1, i2, i3), 1)

            elif index == 3:
                (x1,y1) = kp4[img1_idx].pt
                (x2,y2) = kp1[img2_idx].pt
                cv2.circle(out, (int(x1),int(y1)+rows1), 4, (i1, i2, i3), 1)   
                cv2.circle(out, (int(x2),int(y2)), 4, (i1, i2, i3), 1)
                cv2.line(out, (int(x1),int(y1)+rows1), (int(x2),int(y2)), (i1, i2, i3), 1)



    return out

def drawPoints(img1, kp1, img2, kp2):
   
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for kp in kp1:

        (x1,y1) = kp.pt
        
        i1 = random.randint(0, 255)
        i2 = random.randint(0, 255)
        i3 = random.randint(0, 255)
        
        cv2.circle(out, (int(x1),int(y1)), 4, (i1, i2, i3), 1)   
        
    for kp in kp2:

        (x2,y2) = kp.pt
        
        i1 = random.randint(0, 255)
        i2 = random.randint(0, 255)
        i3 = random.randint(0, 255)
        
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (i1, i2, i3), 1)

    return out

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        
        i1 = random.randint(0, 255)
        i2 = random.randint(0, 255)
        i3 = random.randint(0, 255)
        
        cv2.circle(out, (int(x1),int(y1)), 4, (i1, i2, i3), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (i1, i2, i3), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (i1, i2, i3), 1)


    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

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