import cv2,os
import numpy as np
import csv
import glob
label="Uninfected"
dirList=glob.glob('F://MINI-PROJECT//cell-images-for-detecting-malaria//cell_images//'+label+'//*.png')
file=open("F://MINI-PROJECT//csv//dataset.csv","a+")
for img_path in dirList:
    im=cv2.imread(img_path)
    #syntax: GaussianBlur(source_image,kernel_size,sigma_x)
    im=cv2.GaussianBlur(im,(5,5),2)
    #syntax: cvtColor(source_image,color_code)
    im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #sets to max val(used in Segmentation,syntax:threshold(img,threshold,maxval,type_of_thresholding)
    ret,thresh=cv2.threshold(im_gray,127,255,0)
    contours,_=cv2.findContours(thresh,1,2)
    #syntax: findContours(img,contour_ret_mode,contour_appx_method)
    for contour in contours:
        cv2.drawContours(im_gray,contours,-1,(0,255,0),3)
        #syntax: drawContours(img,contours_list,draw_All_contours,color etc)
        cv2.imshow("window",im_gray)
        break
    file.write(label)
    file.write(",")
    for i in range(5):
        try:
            area=cv2.contourArea(contours[i])
            file.write(str(area))
        except:
            file.write("0")
        file.write(",")
    file.write("\n")
cv2.waitKey(19000)
