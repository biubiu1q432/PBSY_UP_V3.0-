import cv2
import os
import numpy as np
import random


img_dir = r"photo"


class CV_Task:
    
    def __init__(self):
        self.pre = PreProcess()
        self.pr = PointsRec()
        self.c = ColorRec()  

        self.IsEmpty = None
        self.Color = None
        self.Shape = None
    #一次检测
    def Once_cv_task(self,src_img):

        dst = self.pre.pre_process_th(src_img)
        ret,roi,rgb_img = self.pre.Get_roi_and_cut(dst,src_img,debug=0)

        if ret:          
            exp_frame,rgb_img2 = self.pre.Get_edge_and_expand(roi,rgb_img)
            ret,A,B,C,D,shape=self.pr.Get_points(exp_frame,debug=0)
            
            self.Shape = shape

            if ret:
                self.Color,self.IsEmpty=self.c.Color_Recg(rgb_img2,A,B,C,D,debug=0)
            
                print("-------------------------------------")
                print("shape:",shape)
                print("color:",self.Color)
                print("IsEmpty:",self.IsEmpty)
                print("-------------------------------------")


'''预处理工具'''
class PreProcess:
    def __init__(self):

        self.img_x_max = None
        self.img_y_max = None

        self.canny_up = 30
        self.canny_down = 20
        self.expand = 20
         
    #二值化
    def pre_process_th(self,img):
        
        #高斯滤波--》imgBlur
        imgBlur = cv2.GaussianBlur(img, (3, 3), 1)
        #转灰度--》imgGray
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        
        imgGray= imgGray.astype(np.uint8)
        ret,th= cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)     

        self.img_y_max = th.shape[0]
        self.img_x_max = th.shape[1]
    
        return th

    #去白边
    def Get_roi_and_cut(self,th,src_img,step=8,debug = 0):
        

        LEFT = None
        RIGHT = None
        
        l_x = 2
        mid_x = int((self.img_x_max-4)/2)
        r_x = self.img_x_max-2

        debug_img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

        #从左往右
        for x_step in range(l_x,mid_x,step):
                    
            slice_img = th[:,x_step]

            noisy_=np.sum(slice_img == 255)
            density_noisy = noisy_ / self.img_y_max
            
            if debug:
                print(density_noisy)
                a = (x_step,0)
                b = (x_step,self.img_y_max)
                cv2.line(debug_img, a,b,(255,0,0),3)
                cv2.imshow("debug_img", debug_img)
                cv2.waitKey(100)

            if density_noisy <= 0.01:
                LEFT = x_step
                print("@@@@@@@@去白边@@@@@@@@@@")
                print("ROI左边界",LEFT)
                print("左边界密度",density_noisy)
                break

        #从右往左
        for x_step in range(r_x,mid_x,-step):

            slice_img = th[:,x_step]

            noisy_=np.sum(slice_img == 255)
            density_noisy = noisy_ / self.img_y_max
            
            if debug:
                print(density_noisy)
                a = (x_step,0)
                b = (x_step,self.img_y_max)
                cv2.line(debug_img, a,b,(255,255,0),3)
                cv2.imshow("debug_img", debug_img)
                cv2.waitKey(100)

            if density_noisy <= 0.01:
                RIGHT = x_step
                print("ROI右边界",RIGHT)
                print("右边界密度",density_noisy)
                break
        
        if LEFT == None or RIGHT == None or LEFT >= RIGHT:
            print("ROI_error,二值化一片白,返回原图,Get_roi（）不出来")
            print("@@@@@@@@去白边@@@@@@@@@@")
            print()
            
            return False,th,src_img
        else:
            print("ROI区域",LEFT,RIGHT)
            print("@@@@@@@@去白边@@@@@@@@@@")
            print()
            
            roi_img = th[:,LEFT:RIGHT]
            rgb_img = src_img[:,LEFT:RIGHT]
            return True,roi_img,rgb_img

    #边缘化+扩黑边
    def Get_edge_and_expand(self,img,src_img):
        
        self.src_img_height = img.shape[0]
        self.src_img_width = img.shape[1]
        
        imgCanny = cv2.Canny(img, self.canny_down, self.canny_up)
        
        #膨胀
        kernel = np.ones((3, 3), np.uint8)
        imgDilation = cv2.dilate(imgCanny, kernel, iterations=3)
        
        #扩边
        exp_frame = cv2.copyMakeBorder(imgDilation,self.expand,self.expand,self.expand,self.expand,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        rgb_img = cv2.copyMakeBorder(src_img,self.expand,self.expand,self.expand,self.expand,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        self.exp_img_height = exp_frame.shape[0]
        self.exp_img_width = exp_frame.shape[1]

        return exp_frame,rgb_img

'''角点提取'''
class PointsRec:
    
    def __init__(self):

        self.areaMin = 4000
        self.rec_min = 0.7
        self.rec_max = 1.3

    def Get_points(self,img,debug=0):
        
        img_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ret = False
        A,B,C,D = None,None,None,None
        shape = None

        
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)              

            if debug:
                x, y ,w,h = cv2.boundingRect(approx)   
                cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print("当前区域像素点个数",area)
                if area <= self.areaMin:
                    print("不符合面积条件")
                cv2.imshow("img_debug", img_debug)
                cv2.waitKey(0)

            if area > self.areaMin:
                                            
                if debug:
                    print()
                    print("符合面积条件,当前区域角点个数",len(approx))
                    print("分别是：",approx)
                #过滤
                if len(approx) != 4:
                    print("不符合角点个数条件")
                    continue

                A,B,C,D = self.util_points_sort(approx)
                AB,CD,AC = self.util_points_dis(A,B,C,D)
                ret = True

                if debug:
                    cv2.circle(img_debug, (A[0], A[1]), 8, (0, 0, 255), -1)
                    cv2.circle(img_debug, (B[0], B[1]), 8, (0, 0, 255), -1)
                    cv2.circle(img_debug, (C[0], C[1]), 8, (0, 0, 255), -1)
                    cv2.circle(img_debug, (D[0], D[1]), 8, (0, 0, 255), -1)
                    cv2.drawContours(img_debug, [approx], 0, (255, 0, 255), 4)
                    print()
                    print("当前比例AB/CD:",AB/CD)
                    print()
                    cv2.imshow("img_debug", img_debug)
                    cv2.waitKey(0)
                
                if AB/CD < 0.75:
                    print("上梯形")
                    shape = "上梯形"
                elif AB/CD > 1.3:#
                    print("下梯形")
                    shape = "下梯形"
                elif (AB/AC) >= self.rec_max or (AB/AC) <= self.rec_min:
                    print("矩形")
                    shape = "矩形"
                else:
                    print("正方形")
                    shape = "正方形"
                print()

        if ret is not True:
            print("Get_points检测不出,检查roi和过滤条件")
        
        return ret,A,B,C,D,shape
                
    def util_points_sort(self,approx):
        #形状判断
        approx = np.array(approx)
        approx = np.squeeze(approx, axis=1)
        sorted_points = approx[np.argsort(approx[:, 1],)] #按y值从小到大排序

        #------->x
        #|   A   B
        #|   C   D
        #y
        
        #AB
        if sorted_points[0][0] <= sorted_points[1][0]:
            A = sorted_points[0]
            B = sorted_points[1]
        elif sorted_points[0][0] > sorted_points[1][0]:
            A = sorted_points[1]
            B = sorted_points[0]
        #CD
        if sorted_points[2][0] <= sorted_points[3][0]:
            C = sorted_points[2]
            D = sorted_points[3]
        elif sorted_points[2][0] >= sorted_points[3][0]:
            C = sorted_points[3]
            D = sorted_points[2]

        return A,B,C,D

    def util_points_dis(self,A,B,C,D):
        AB = int(np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2))#AB距离计算               
        CD = int(np.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)) #CD距离计算
        AC = int(np.sqrt((A[0]-C[0])**2 + (A[1]-C[1])**2))#AC距离计算
        return AB,CD,AC
    
'''颜色识别'''
class ColorRec:
    
    def __init__(self):
        self.roi_A_shift = 25   #A点偏移量
        self.roi_range = 15 
        
        self.white_bgr_lower= np.array([0, 0, 77])
        self.white_bgr_upper= np.array([136 ,119 ,243])     
        self.white_ite = 2

        self.red_bgr_lower= np.array([0 ,53, 83])
        self.red_bgr_upper= np.array([6 ,255, 255])     
        self.red_ite = 2

        self.blue_bgr_lower= np.array([89, 65 ,77])
        self.blue_bgr_upper= np.array([154, 255, 166])     
        self.blue_ite = 2

    '''
    color:颜色
    is_empty:是否空心
    '''
    def Color_Recg(self,rgb_img2,A,B,C,D,debug=0):
        
        
        #中心roi
        center_x,center_y,roi_x,roi_y = self.util_roi_center(A,B,C,D,rgb_img2,debug)
        center_roi = self.roi_get(rgb_img2,center_x,center_y,debug)
    
        #判断是否空心      
        ret = self.judge_center_isempty(center_roi)
        
        #空心
        if ret:
            #左上roi
            ul_roi = self.roi_get(rgb_img2,roi_x,roi_y,debug)
            color  = self.color_get(ul_roi)
            is_empty = True
        #实心
        else:
            color = self.color_get(center_roi)
            is_empty = False
        
        return color,is_empty

    def color_get(self,roi_img):
        
        
        hsv_image_red = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image_red, self.red_bgr_lower, self.red_bgr_upper)
        red_th = cv2.erode(red_mask, np.ones((3,3),np.uint8), iterations=self.red_ite)
        red_count = np.sum(red_th == 255)

        hsv_image_blue = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image_blue, self.white_bgr_lower, self.white_bgr_upper)
        blue_th = cv2.erode(blue_mask, np.ones((3,3),np.uint8), iterations=self.blue_ite)
        blue_count = np.sum(blue_th == 255)

        if red_count > blue_count:
            print("红色")
            return "红色"
        else:
            print("蓝色")
            return "蓝色"

    ''' 
    @breif:判断中心区域是否为空心，并识别颜色
    @ret:True为空心，False为实心
    @color:1红色 2蓝色
    '''
    def judge_center_isempty(self,center_roi,debug = 0):
        
        #颜色空间转换
        hsv_image = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

        #二值转换，进行颜色分割---》把色域内的像素点设为白色，其余像素点设为黑色
        lower_color = self.white_bgr_lower
        upper_color = self.white_bgr_upper
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        #开运算
        th = cv2.erode(mask, np.ones((3,3),np.uint8), iterations=self.white_ite)

        #在th中统计值为255的像素点的个数
        count_target = np.sum(th == 255)
        count_else = np.sum(th == 0)
        
        if debug:
            
            print()
            print("&&&&&&judge_center_isempty&&&&&&&&")
            print("count_target:",count_target)
            print("count_else:",count_else)
            print("&&&&&&judge_center_isempty&&&&&&&&")
            print()


        
        if count_target > count_else:
            if debug:
                print("空心")
                print()

            return True
        else:
            if debug:
                print("实心")
                print()
            return False

    '''获取中心区域roi'''
    def roi_get(self,src_img,center_x,center_y,debug = 0):
        #取以center_x,center_y为中心，roi_range为半径的roi
        center_roi = src_img[center_y-self.roi_range:center_y+self.roi_range,center_x-self.roi_range:center_x+self.roi_range]
        
        if debug:
            debug_roi_img = src_img
            cv2.rectangle(debug_roi_img,(center_x-self.roi_range,center_y-self.roi_range),(center_x+self.roi_range,center_y+self.roi_range),(255,0,0),2)
            cv2.imshow("debug_roi_img", debug_roi_img)
            cv2.waitKey(0)

        return center_roi

    def util_roi_center(self,A,B,C,D,src_img,debug = 0):
        center_x = int((A[0]+B[0]+C[0]+D[0])/4)
        center_y = int((A[1]+B[1]+C[1]+D[1])/4)

        roi_x = int(A[0]+self.roi_A_shift)
        roi_y = int(A[1]+self.roi_A_shift)
        
        if debug:
            debug_img = src_img
            cv2.circle(debug_img,(center_x,center_y),5,(255,0,255),-1)
            cv2.circle(debug_img,(roi_x,roi_y),5,(0,0,255),-1)
            cv2.imshow("debug_roi_get_point", debug_img)
            cv2.waitKey(0)
        
        return center_x,center_y,roi_x,roi_y


def debugger():
    pre = PreProcess()
    pr = PointsRec()
    c = ColorRec()  

    for i in range(1, 55):

        #读图
        img_name = str(i)+".jpg"
        img_path = os.path.join(img_dir, img_name)
        src_img = cv2.imread(img_path)
        cv2.imshow("src_img", src_img)    
        cv2.waitKey(0)

        #预处理
        dst = pre.pre_process_th(src_img)
        ret,roi,rgb_img = pre.Get_roi_and_cut(dst,src_img,debug=1)

        if ret:
            
            #角点提取
            exp_frame,rgb_img2 = pre.Get_edge_and_expand(roi,rgb_img)
            cv2.imshow("exp_frame", exp_frame)
            cv2.waitKey(0)
            ret,A,B,C,D=pr.Get_points(exp_frame,debug=1)
            
            if ret:
                c.Color_Recg(rgb_img2,A,B,C,D)



if __name__ == '__main__':
    # debugger()
    
    indx = random.randint(1, len(os.listdir(img_dir)))
    img_name = str(indx)+".jpg"
    img_path = os.path.join(img_dir, img_name)
    
    print(img_path)
    
    print()
    src_img=cv2.imread(img_path)
    cvt = CV_Task()
    cvt.Once_cv_task(src_img)
    


    