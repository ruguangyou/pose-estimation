#!/usr/bin/python3

import cv2 as cv
import glob
import os
images = glob.glob('*.jpg')
try:
    os.mkdir('left')
    os.mkdir('right')
except:
    for fname in images:
        img = cv.imread(fname)
        (rows, cols) = img.shape[0:2]
        # left lens
        left = img[0:rows, 0:cols//2]
        # right lens
        right = img[0:rows, cols//2:cols]
    
        os.chdir('left')
        imgname = fname[:-4] + '_left.jpg' # fname is *.jpg
        cv.imwrite(imgname, left)
    
        os.chdir('../right')
        imgname = fname[:-4] + '_right.jpg'
        cv.imwrite(imgname, right)
        os.chdir('..')
