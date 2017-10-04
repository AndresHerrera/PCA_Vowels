#############################################
#### PCA for vocals recognition
# Basado en los ejemplos de curso Fundamentos de Sistemas Distribuidos
# Docente : Maria Patricia - maria.trujillo@correounivalle.edu.co 
# Course: INTRODUCTION TO PATTERN RECOGNITION FOR COMPUTER VISION-01
# Author:  Fabio Andres Herrera - fabio.herrera@correounivalle.edu.co
################################################
## File: extract_vowels.py
## Requirements :  
## - OpenCV 3.2.0
## - NumPy 1.11.3           
################################################
##  Extract and crop single vowels images from an array image
##  Algorithm Steps : 
##  -  Read Training Image
##  -  Image Segmentation
##  -  Find Contours
##  -  Crop image into pieces
##  -  Resize pieces
##  -  Save into a sigle files 
#################################################


import os
import cv2 
import numpy as np

w = 20
h = 20
min_contour_ara = 100


#Prepare Image Dataset

def main():
    imgVowels = cv2.imread("raw_vowels.jpg")   # read in training vowels image

    imgGray = cv2.cvtColor(imgVowels, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) 


    cv2.imshow("Threshold Image", imgThresh)      # show threshold image for reference

    imgThreshCopy = imgThresh.copy()

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)          
    
    found=0
    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > min_contour_ara:          # if contour is big enough to consider
        	[intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

            # draw rectangle around each contour
           	cv2.rectangle(imgVowels,(intX, intY),(intX+intW,intY+intH),(0, 0, 255),2) 

           	imgROI = imgThresh[intY:intY+intH, intX:intX+intW]  # crop char out of threshold image
           	imgROIResized = cv2.resize(imgROI, (w, h))     # resize image

           	cv2.imshow("Original File", imgVowels)      # show training vowels image

     		found+=1

           	cv2.imwrite('vowels/vowel_'+str(found)+'.png',cv2.bitwise_not(imgROIResized))

    print str(found) + " files written into (vowels) folder !"
    print "Notice: You have to arrange (vowels) files into (training) folder tree !"
    print "in folder :  training/A  <-  store only A images"
    print "in folder :  training/E  <-  store only E images"
    print "in folder :  training/I  <-  store only I images"
    print "in folder :  training/O  <-  store only O images"
    print "in folder :  training/U  <-  store only U images"

    cv2.waitKey(0)
    cv2.destroyAllWindows()

###################################################################################################
if __name__ == "__main__":
    main()