
############################################################################
#颜色二值化阈值调试
############################################################################

import cv2
import numpy as np
import os
import keyboard
import time
start,end = 0.0,0.0
def nothing(x):
    pass
WindowName = 'result'
cv2.namedWindow(WindowName, cv2.WINDOW_KEEPRATIO)  # 建立空窗口
cv2.resizeWindow(WindowName, 200, 160)  # 调整窗口大小
cv2.createTrackbar('Bl', WindowName, 53, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Gl', WindowName, 15, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Rl', WindowName, 119, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Bh', WindowName, 141, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Gh', WindowName, 174, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Rh', WindowName, 217, 255, nothing)  # 创建滑动条
cv2.createTrackbar('iterations', WindowName, 2, 20, nothing)  # 创建滑动条


class Color_debugger:
     
     def __init__(self,pic_dir):
        self.root_dir = pic_dir
        
     
     def for_cap(self):
        
        cap =cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        while True:            
            ret,frame = cap.read()
            if ret:
                
                # 获取滑动条值
                Bl = cv2.getTrackbarPos('Bl', WindowName)  
                Gl = cv2.getTrackbarPos('Gl', WindowName)  
                Rl = cv2.getTrackbarPos('Rl', WindowName)

                Bh = cv2.getTrackbarPos('Bh', WindowName)
                Gh = cv2.getTrackbarPos('Gh', WindowName)
                Rh = cv2.getTrackbarPos('Rh', WindowName)

                ite = cv2.getTrackbarPos('iterations', WindowName)

                #颜色空间转换
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
                #二值转换，进行颜色分割---》把色域内的像素点设为白色，其余像素点设为黑色
                lower_color = np.array([Bl, Gl, Rl])
                upper_color = np.array([Bh, Gh, Rh])
                mask = cv2.inRange(hsv_image, lower_color, upper_color)

                #开运算
                hsv_image = cv2.erode(mask, np.ones((3,3),np.uint8), iterations=ite)

                cv2.imshow(WindowName, hsv_image)
                if cv2.waitKey(1) == ord('q'):
                    break

     def for_pic(self):
        pci_dir=os.listdir(self.root_dir)
        pci_num =len(pci_dir)
        
        for i in range(pci_num):
            

            while True:
            
                img = cv2.imread(os.path.join(self.root_dir,pci_dir[i]))
                # 获取滑动条值
                self.Bl = cv2.getTrackbarPos('Bl', WindowName)
                self.Gl = cv2.getTrackbarPos('Gl', WindowName)
                self.Rl = cv2.getTrackbarPos('Rl', WindowName)

                self.Bh = cv2.getTrackbarPos('Bh', WindowName)
                self.Gh = cv2.getTrackbarPos('Gh', WindowName)
                self.Rh = cv2.getTrackbarPos('Rh', WindowName)

                self.ite = cv2.getTrackbarPos('iterations', WindowName)

                #颜色空间转换
                hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                #二值转换，进行颜色分割---》把色域内的像素点设为白色，其余像素点设为黑色
                lower_color = np.array([self.Bl, self.Gl, self.Rl])
                upper_color = np.array([self.Bh, self.Gh, self.Rh])
                mask = cv2.inRange(hsv_image, lower_color, upper_color)

                #开运算
                hsv_image = cv2.erode(mask, np.ones((3,3),np.uint8), iterations=self.ite)

                cv2.imshow(WindowName, hsv_image)
                cv2.imshow('src',img)
                
                #如果键盘按下f
                if keyboard.is_pressed('f'):
                    #用户输入字符串
                    color_type = input("请输入当前颜色：")
                    print("Bl, Gl, Rl ",self.Bl, self.Gl,self.Rl)
                    print("Bh, Gh, Rh ", self.Bh, self.Gh, self.Rh)
                    print("iterations ",self.ite)
                    #把结果写入txt文件
                    with open(r"color_ret.txt", 'a') as f:
                        f.write(color_type + "\n")
                        f.write("Bl, Gl, Rl : " + str(self.Bl) + " ," + str(self.Gl) + ", " + str(self.Rl) + "\n")
                        f.write("Bh, Gh, Rh : " + str(self.Bh) + " ," + str(self.Gh) + ", " + str(self.Rh) + "\n")
                        f.write("iterations : " + str(self.ite) + "\n")
                        f.write("\n")
                        cv2.waitKey(10)
                
                if cv2.waitKey(1) == ord('q'):
                    break
        


cd = Color_debugger(r"D:\CV\ROBOCUP\photo")
cd.for_pic()