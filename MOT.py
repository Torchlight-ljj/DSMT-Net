import os
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def find_inner_circle(imgpath):
    '''
    Find the top locating circle.
    Return to Center of circle x, Center of circle y, Radius
    '''

    def find_inner_circle_var_median(ksize):
        '''
        By setting different median filter size values
        Return the circle set
        '''
        # Read the file
        img = cv.imdecode(np.fromfile(imgpath), cv.IMREAD_UNCHANGED) # Chinese filename support

        # Top and bottom cover
        # TODO: Can be removed, if the threshold can be made good
        width, height = img.shape[:2]
        imgdetect = cv.rectangle(img, (0,0),(int(1.5*width),int(0.08*height)),(0,0,0),-1)
        imgdetect = cv.rectangle(imgdetect, (int(1.5*width),height), (0,int(0.5*height)),(0,0,0),-1)
        imgdetect = cv.rectangle(imgdetect, (int(0.3*width),height), (0,0),(0,0,0),-1)

        # Median filtering
        imgblur = cv.medianBlur(img, ksize)

        # Grayscale images
        imgray = cv.cvtColor(imgblur, cv.COLOR_BGR2GRAY) 

        # Canny Finding the edge
        edges = cv.Canny(imgray,50,150,apertureSize=3)

        # Hough circle transformation
        circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

        return circles

    ksize_var = 9               # Median filter starting value

    # Iterate until the circle is found
    circles = find_inner_circle_var_median(ksize_var)
    while not (str(type(circles))) == "<class 'numpy.ndarray'>":
        ksize_var -= 2
        if ksize_var < 1: 
            raise Exception("No circle is found!")  # Exception: circle not found
        circles = find_inner_circle_var_median(ksize_var)

    # Re-read the file
    img = cv.imdecode(np.fromfile(imgpath), cv.IMREAD_UNCHANGED) # Chinese filename support

    circles = np.uint16(np.around(circles))

    # # Output all circles
    # for i in circles[0,:]:
    #     # draw the outer circle
    #     cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv.circle(img,(i[0],i[1]),2,(0,0,255),3)

    # Topmost circle
    topcircle = circles[:,np.argmin(circles[:,:,1])][0,:]
    # draw the outer circle
    cv.circle(img,(topcircle[0],topcircle[1]),topcircle[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img,(topcircle[0],topcircle[1]),2,(0,0,255),3)

    return topcircle[0], topcircle[1], topcircle[2]

def find_border(imgpath):
    '''
    Gets the body boundary information.
    Returns the leftmost point, the rightmost point and the lowest point.
    '''
    # read file
    img = cv.imdecode(np.fromfile(imgpath), cv.IMREAD_GRAYSCALE) # Chinese filename support

    # Median filtering
    img = cv.medianBlur(img, 5)

    # Adaptive thresholds
    thresh = cv.threshold(img, 30, 255, cv.THRESH_BINARY)[1]
    
    # Finding the edge
    contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    # Find the single edge with the largest area
    cntAreas = np.fromiter((cv.contourArea(cnt) for cnt in contours), float)
    outline = contours[np.argmax(cntAreas)]

    leftmost    = tuple(outline[outline[:,:,0].argmin()][0])
    rightmost   = tuple(outline[outline[:,:,0].argmax()][0])
    bottommost  = tuple(outline[outline[:,:,1].argmax()][0])

    return leftmost, rightmost, bottommost

def getmeta(imgpath):
    '''
    Gets the range of sector circles scanned by the image.
    Returns the centre, radius and angle of the circle.
    '''
    cx, cy, inner_radius = find_inner_circle(imgpath)
    l, r, b = find_border(imgpath)
    outer_radius = round((max(abs(cx-l[0]), abs(r[0]-cx)) + abs(b[1]-cy)) / 2)
    center = (cx, cy)
    radius = (inner_radius, outer_radius)
    theta  = (30, 150)
    return center, radius, theta

def polar(I, center, r, theta=(0,360), rstep=1,thetastep=360.0/(180*8)):
    '''
    Polar coordinate transformation
    '''
    h,w = I.shape[:2]                               # height and width of image
    cx,cy = center                                  # Coordinates of the centre of the circle
    minr, maxr = r                                  # Input image radius range
    mintheta, maxtheta = theta                      # Input image angle range
    H = int(abs(maxr - minr) / rstep) + 1              # Output image height
    W = int(abs(maxtheta - mintheta) / thetastep) + 1  # Output image weight
    O = 125 * np.ones((H, W) , I.dtype)             # Output image space


    r = np.linspace(minr, maxr, H)
    r = np.tile(r, (W,1))
    r = np.transpose(r)                             # Equally divided in height direction
    theta = np.linspace(mintheta, maxtheta, W)
    theta = np.tile(theta, (H,1))                   # Equally divided in the width direction
    x,y = cv.polarToCart(r, theta, angleInDegrees=True)  # Mapping coordinates

    # Nearest Neighbour Interpolation
    for i in range(H):
        for j in range(W):
            px = int(round(x[i][j])+cx)
            py = int(round(y[i][j])+cy)
            if((px >= 0 and px <= w-1) and (py >= 0 and py <= h-1)):
                O[i][j] = I[py][px]
            else:
                O[i][j] = 125

    O = cv.flip(O,1)                                # Flip the image horizontally
    
    return O

#########################Start Analysis##########################

inputfile = './test.jpg'
outputfile = './test_MOT.jpg'
center, radius, theta = getmeta(inputfile)
inputimg = cv.imdecode(np.fromfile(inputfile), cv.IMREAD_GRAYSCALE)
outputimg = polar(inputimg, center, radius, theta)
cv.imwrite(outputfile,outputimg)
