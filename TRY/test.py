
import os
import threading
import cv2
import numpy as np
img_dir = r"ROBOCUP\photo"

#设置滑动条
cv2.namedWindow('original_img')
cv2.createTrackbar('canny1', 'original_img', 22, 255, lambda x: None)
cv2.createTrackbar('canny2', 'original_img',  30, 255, lambda x: None)
cv2.createTrackbar('bf_sc', 'original_img',  0, 255, lambda x: None)
cv2.createTrackbar('bf_ss', 'original_img',  0, 255, lambda x: None)
if __name__ == '__main__':
        for i in range(1, 55):
                
                while True:
                        #读图
                        img_name = str(i)+".jpg"
                        img_path = os.path.join(img_dir, img_name)
                        src_img = cv2.imread(img_path)

                        #去光照
                        dist = cv2.bilateralFilter(src_img, 5, 200, 200)        
                        h,s,v = cv2.split(dist)
                        v = cv2.blur(v,(11,11))
                        dist = cv2.merge([h,s,v])

                        #二值化
                        gray = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)
                        dist = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
                        
                        
                        p1 = cv2.getTrackbarPos('canny1', 'original_img')
                        p2 = cv2.getTrackbarPos('canny2', 'original_img')
                        edge = cv2.Canny(dist, p1, p2)
                                                                               
                        cv2.imshow('original_img',src_img)
                        cv2.imshow('dsit',dist)
                        cv2.imshow('edge',edge)
                
                        if cv2.waitKey(100) & 0xFF == ord('q'):
                                break


