# -------------------------------------------------------------------- #





#                          Image Processing


#                      Elastic Active Materials





# 1) Hough transform with fixed radius to determine the location of the annulus


# 2-0) Find springs directions


# 2-1) Iterate on all the found circles


# - Extract the small square image in which the hexbug can be found


# - Binarize it (contour or image itself)


# - Extract orientation from image moments


# - Find directions





# -------------------------------------------------------------------- #





# -------------------------------------------------------------------- #





# Hough circle transform parameters :





# dp - The inverse ratio of resolution


# minDist - Minimum distance between detected centers


# param1 - Upper threshold for the internal Canny edge detector


# param2 - Threshold for center detection.
# -------------------------------------------------------------------- #

# Imports
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import scipy.stats
from scipy import ndimage
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import imutils
import time
import csv
import sys
import argparse
import os
from glob import glob
from pprint import pprint
import re

"""
    Set globals
"""
# Goal parameters
Nframe = 1
numb = 4

# General algorithm parameters
r = 65            # Radius of detected circles (in pixels)
minDist = 95      # Minimum distance accepted between two detected circles
param1 = 27       # Parameter 1 for hough transform
param2 = 32       # Parameter 2 for hough transform
delta = 5         # Accepted variance for radius of detected circles
springsWidth = 20 # For annulus orientation (in pixels)
springsDepth = 5  # For annulus orientation (in pixels)
sigma_ = 0.2      # Accepted variance for the auto-edge detection (smaller sigma means tighter edge detection)
gaussianFilterParam_ = 5 # Gaussian filter size before contour detection (must be odd)

# Get files here
def getFiles(dname):
    print(f"Loading from Images/{dname}")
    files=glob(f"../Images/{dname}/img*.png")
    files=sorted(files,key=lambda f: int(re.findall('\d+',os.path.split(f)[-1])[0]))
    return files

# Subsidiary functions
def printCircle(x0,y0,r):
    theta = np.linspace(0,2*np.pi,50)
    x = r*np.cos(theta) + x0
    y = r*np.sin(theta) + y0
    plt.plot(x,y,'k-', linewidth=0.5)

def printDashedCircle(x0,y0,r):
    theta = np.linspace(0,2*np.pi,50)
    x = r*np.cos(theta) + x0
    y = r*np.sin(theta) + y0
    plt.plot(x,y,'k--', linewidth=2.0)

def moment(img, p, q):
    (ni, nj) = img.shape
    tmp = np.array([[(j**p)*(i**q) for i in range(ni)] for j in range(nj)], dtype=float)*img
    return np.sum(tmp)

def rotate(l, n):
    return np.concatenate((l[n:],l[:n]),axis=0)

def name(i,digit=4):
    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = i+'.jpg'
    return i

def sortByClosest(X_, Y_, X, Y):
    order = []
    for i in range(len(X)):
        dist = []
        for j in range(len(X_)):
            dist.append(np.sqrt((X[i] - X_[j])**2 + (Y[i] - Y_[j])**2))
        order.append(dist.index(np.min(dist)))
    return order

def TicTocGenerator():
    """
    Generator that returns time differences
    """
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

def toc(tempBool=True):
    """
    Prints the time difference yielded by generator instance TicToc
    """
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("-> Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    """
    Records a time in TicToc, marks the beginning of a time interval
    """
    toc(False)

def GraysCut(image, ref):
    img_return = (image>ref)*ref + (image<=ref)*image
    img_return = np.array(img_return, dtype=np.uint8)
    return img_return

def detect_SI(gray, r, minDist, param1, param2, delta, gaussianFilterParam_):
    """Detect circles via Hough transform"""
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=minDist,
        param1=param1, param2=param2, minRadius=r-delta,
        maxRadius=r+delta
        )
    print(str(len(circles[0]))+' Circles detected')
    print('')
    C = np.round(circles[0, :]).astype("int")
    #print(C)
    X = []
    Y = []
    N = []
    Alpha = []
    insideAnnulus = {}
    insideAnnulus_Bin = {}
    if C is not None:
        for i in range(len(C)):
            x, y, r = C[i][0], C[i][1], C[i][2]
            X.append(x)
            Y.append(y)
            bug_ = np.copy(gray[y-r:y+r, x-r:x+r])
            # Pedagogical Output
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(bug_[::-1,:],cmap='gray')
            printDashedCircle(r, r, r-delta)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0,2*r])
            ax.set_ylim([0,2*r])
            ax.axis('off')
            plt.savefig(f'../Images/Checks/{args.exp_name}/insideAnnulus{i}.png',dpi=300)
            plt.close(fig)

            # Exclude what is not in the detected circle
            for m in range(2*r):
                for n in range(2*r):
                    if np.sqrt((m-r)**2 + (n-r)**2) > r:
                        bug_[m,n] = 255

            # Thresholding
            sorted4Bin = np.sort(bug_, axis=None)
            threshold4Bin = sorted4Bin[2000] # We know the area of the bug
            blurred = cv2.GaussianBlur(bug_, (gaussianFilterParam_, gaussianFilterParam_), 1) # Not always necessary
            retval, bugBin_ = cv2.threshold(blurred, threshold4Bin, 255, cv2.THRESH_BINARY_INV)

            # Clean the edges of the annulus
            for m in range(2*r):
                for n in range(2*r):
                    if np.sqrt((m-r)**2 + (n-r)**2) > r-delta:
                        bugBin_[m,n] = 0
            insideAnnulus[i] = bug_
            insideAnnulus_Bin[i] = bugBin_
            # Compute the moment to extract orientation
            #mmts = cv2.moments(edges_) # More costly method
            m20 = moment(bugBin_, 2, 0)
            m02 = moment(bugBin_, 0, 2)
            m11 = moment(bugBin_, 1, 1)
            m10 = moment(bugBin_, 1, 0)
            m01 = moment(bugBin_, 0, 1)
            m00 = moment(bugBin_, 0, 0)
            if m00 != 0:
                mu20 = m20/m00 - (m10/m00)**2
                mu02 = m02/m00 - (m01/m00)**2
                mu11 = m11/m00 - (m01*m10)/(m00**2)
                theta = 0.5*np.arctan2(2*mu11,(mu20-mu02))
                N.append(theta)
            else:
                N.append(float('nan'))

            # Pedagogical Output
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            x_tmp = np.linspace(0,2*r,10)
            y_tmp = x_tmp*np.tan(theta-np.pi/2) + 2*r - m10/m00 - m01*np.tan(theta-np.pi/2)/m00
            ax.imshow(bugBin_[::-1,:],cmap='binary')
            ax.plot([m01/m00], [2*r-m10/m00], 'ro', markersize=9.0)
            ax.plot(x_tmp, y_tmp, 'r-', linewidth=4.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0,2*r])
            ax.set_ylim([0,2*r])
            ax.axis('off')
            plt.savefig(f'../Images/Checks/{args.exp_name}/insideAnnulus_Bin{i}.png',dpi=300)
            plt.close(fig)

            fig = plt.figure(figsize=(2,2))
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(bug_[::-1,:],cmap='gray')
            ax.plot([m01/m00], [2*r-m10/m00], 'ro', markersize=9.0)
            ax.plot(x_tmp, y_tmp, 'r-', linewidth=4.0)
            printDashedCircle(r, r, r-delta)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0,2*r])
            ax.set_ylim([0,2*r])
            ax.axis('off')
            plt.savefig(f'../Images/Checks/{args.exp_name}/{i}_bug_in_circle.png',dpi=300)
            plt.close(fig)
    else:
        print("No circles")
    return X,Y,N

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Image processing code')
    parser.add_argument('exp_name',action='store',type=str,
                        help='name of the experiment label')
    args = parser.parse_args()
    files=getFiles(args.exp_name)

    # Set up Output directories
    os.makedirs(f"../Data/{args.exp_name}/img_processed",exist_ok=True)
    os.makedirs(f"../Images/Check/{args.exp_name}",exist_ok=True)



    ###############################################################################
    TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
    tic()
    # Load the images
    Xt = {}
    Yt = {}
    Nt = {}
    ref = 120
    for im,file in enumerate(files):
        print(f'--{file}--')
        image=cv2.imread(file)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray=GraysCut(gray, ref)
        X_,Y_,N_=detect_SI(gray,r,minDist,param1,param2,delta,gaussianFilterParam_)
        try_P=1

        while len(X_) > numb and try_P < 20: # Modify treshold to find the right number of annulus
            param2 = param2 + 1
            X_, Y_, N_ = detect_SI(gray, r, minDist, param1, param2, delta, gaussianFilterParam_)
            try_P += 1
        try_M = 1
        while len(X_) < numb and try_P < 20: # Modify treshold to find the right number of annulus
            param2 = param2 - 1
            X_, Y_, N_ = detect_SI(gray, r, minDist, param1, param2, delta, gaussianFilterParam_)
            try_M += 1
        #print(X_, Y_, N_)
        if len(X_) != numb: # Catch the writing error if while loops fail
            break
        if im == 0:
            X = np.copy(X_)
            Y = np.copy(Y_)
            N = np.copy(N_)
        if im > 0:
            order = sortByClosest(X_, Y_, Xt[im-1], Yt[im-1])
            X = [X_[i] for i in order]
            Y = [Y_[i] for i in order]
            N = [N_[i] for i in order]
            for i in range(len(N)):
                if abs(N[i]-Nt[im-1][i])>(np.pi/2) and abs(N[i]-Nt[im-1][i])<(3*np.pi/2):
                    if (N[i] - Nt[im-1][i])%(2*np.pi) > 0:
                        N[i] = (N[i] - np.pi)%(2*np.pi)
                    else:
                        N[i] = (N[i] + np.pi)%(2*np.pi)
        i_invert = np.argmin(Y)
        N[i_invert] = N[i_invert] + np.pi
        Xt[im] = X
        Yt[im] = Y
        Nt[im] = N
        # Outputs

        fig = plt.figure()
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        for i in range(len(N_)):
            printCircle(X[i], Y[i], r)
            plt.quiver(X[i], Y[i], -np.sin(N[i]), np.cos(N[i]),
                       pivot='mid', zorder=1, scale=15,
                       color='red', headwidth=3.0)
        plt.savefig(f'../Data/{args.exp_name}/img_processed/{name(im)}', dpi=300)
        plt.close(fig)
    toc()
    # Write the CSV
    # Save the file of neighbors
    fname = "data.csv"
    file = open(fname, "w")
    try:
        writer = csv.writer(file, delimiter=",")
        for i in range(Nframe):
            writer.writerow(Xt[i])
            writer.writerow(Yt[i])
            writer.writerow(Nt[i])
    finally:
        file.close()
