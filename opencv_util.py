# coding=utf-8
#This is the image detecting functions file

import numpy as np
import cv2
import math

class FindObj:
    _targetImg = ""#要找的目标图片
    _regionImg = ""#被搜索的大图
    _original=None#截图时，原始图片的宽高[width,height]
    _region=None
    _ratio=0.75
    _zoom_region=1.0#缩放比率

    def __init__(self, targetImg, regionImg, original,region):
        self._region=region
        self._original = original
        self._targetImg = cv2.imread(targetImg, 0)
        self._regionImg = cv2.imread(regionImg, 0)
        tempH, tempW=self._regionImg.shape[:2]#缩放前的被搜索图片


        #检查是否横屏，对屏幕进行旋转
        if (original and (self._original[0]<self._original[1]) ^ (tempW<tempH)):
            self._regionImg = self.rotate_about_center(self._regionImg,90)
            tempChange = tempH
            tempH = tempW
            tempW = tempChange

        if original:
            self._zoom_region=float(self._original[0])/float(tempW)#获得缩放比率

        if self._zoom_region != 1:
            #如果和原图不一致
            self._regionImg=cv2.resize(self._regionImg,(int(tempW*self._zoom_region),int(tempH*self._zoom_region)),cv2.INTER_CUBIC)
            #对reginImg进行缩放

        if self._region is not None:
            if(not self._region.height()):
                self._region.setX(0)
                self._region.setHeight(tempH)
            if(not self._region.width()):
                self._region.setY(0)
                self._region.setWidth(tempW)
            #检查区域是否有意义，如果没有则默认为全屏
            #print('截取后的坐标')
            #print(self._region.y(),(self._region.height()+self._region.y()), self._region.x(),(self._region.width()+self._region.x()))
            self._regionImg=self._regionImg[self._region.y():(self._region.height()+self._region.y()), self._region.x():(self._region.width()+self._region.x())];
            #按照比例，截取后的图片  cv2.imwrite('/usr/lib/python2.7/site-packages/shot.png',self._regionImg)

    def rotate_about_center(self,src, angle, scale=1.0):
        #中心旋转
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)

        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale

        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)

        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))

        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    def findMiddlePointByAkaze(self):
        detector = cv2.AKAZE_create()
        [regionX,regionY],status = self._findMiddlePoint(detector)
        if(self._region is not None and regionX is not None and regionY is not None):
            return [regionX + self._region.x(), regionY + self._region.y(),status];
        else:
            return [regionX,regionY,status]

    def findMiddlePointByBrisk(self):
        detector = cv2.BRISK_create()
        [regionX,regionY],status,max_val = self._findMiddlePoint(detector)
        #返回中心点坐标。但是图片存在缩放的可能,需要self._zoom_region来修正
        #print(regionX)
        #print(regionY)

        if(self._region is not None and regionX is not None and regionY is not None):
            return [float(regionX + self._region.x())/self._zoom_region, float(regionY + self._region.y())/self._zoom_region,status,max_val];
        elif (regionX is not None and regionY is not None):
            return [float(regionX)/self._zoom_region,float(regionY)/self._zoom_region,status,max_val]
        else:
            return [regionX,regionY,status,max_val]

    def findMiddlePointByOrb(self):
        detector = cv2.ORB_create(400)
        regionX,regionY = self._findMiddlePoint(detector)
        if(self._region is not None and regionX is not None and regionY is not None):
            return [regionX + self._region.x(), regionY + self._region.y()];
        else:
            return [regionX,regionY]

    def findFeatureNumByBrisk(self):
        detector = cv2.BRISK_create()
        if self._targetImg is None or self._regionImg is None:
            return ""
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        kp1, desc1 = detector.detectAndCompute(self._targetImg, None)
        kp2, desc2 = detector.detectAndCompute(self._regionImg, None)
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = self._filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:#computing a matrix nees 4 matched points at least
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            return ('%d / %d' % (np.sum(status), len(status)))
        else:
            H, status = None, None
            return ('< %d ' % len(p1))

    def _findMiddlePoint(self,detector):
        if self._targetImg is None or self._regionImg is None:
            #是否存在相应的图片
            #print('in fall wrong target Img or retionImg')
            #print(self._targetImg)
            #print(self._regionImg)
            return [None,None],None,None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        kp1, desc1 = detector.detectAndCompute(self._targetImg, None)
        kp2, desc2 = detector.detectAndCompute(self._regionImg, None)
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = self._filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:#computing a matrix needs 4 matched points at least
 #           print('this.originNal Widt')
 #           print(self._originalWidth)
            if self._original is not None:
                #判断原始图片是否为空：判断是否为之前的脚本
  #              cv2.imwrite('/usr/lib/python2.7/site-packages/midori.png',self._regionImg)
 #               cv2.imwrite('/usr/lib/python2.7/site-packages/midtar.png',self._targetImg)
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                middlePoint=self._explore(kp_pairs, status, H)
                h1, w1 = self._targetImg.shape[:2]
                h2, w2 = self._regionImg.shape[:2]
                #此时reginImg为经过缩放处理后图片
#middlePoint[0]----x   middlePoint[1]----y
                if(middlePoint[0] is not None and middlePoint[0] > 0 and middlePoint[0]<w2 and middlePoint[1] is not None and middlePoint[1] >0 and middlePoint[1]<h2):
                    #判断中心点在所选区域内
                    #验证中心点坐标两个宽高的范围内，模板匹配的结果
                    #剪切出模板匹配的目标范围图像searchRange

                    #regionImgZoomed=cv2.resize(self._regionImg,(int(w2*self._zoom_region),int(h2*self._zoom_region)),cv2.INTER_CUBIC)
                    #print('@@ hw @@&')
                    #print(h1)
                    #print(w1)
                    #print(h2)
                    #print(w2)
                    #print(self._zoom_region)
                    #print(middlePoint)
                    #print('&@@@@')
                    x_left=middlePoint[0]-w1
                    x_right=middlePoint[0]+w1
                    y_top=middlePoint[1]-h1
                    y_bottom=middlePoint[1]+h1
                    #判断，如果目标图片所在的区域超过了截图区域，则需要修正，保证目标图片存在于区域内部
                    if x_left<0:
                        x_left=0
                    if x_right>w2:
                        x_right=w2
                    if y_top<0:
                        y_top=0
                    if y_bottom > h2:
                        y_bottom = h2
                    searchRange=self._regionImg[y_top:y_bottom, x_left:x_right];
#                        cv2.imwrite('/usr/lib/python2.7/site-packages/shot.png',self._regionImg)

                    if self._zoom_region>1:
                        #如果运行时候的图片是经过放大，则需要进行锐化操作
                        #锐化
                        sharpenedSR=np.zeros(searchRange.shape, np.uint8)
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                        sharpenedSR = cv2.filter2D(searchRange, -1, kernel)
                        #中值滤波，降噪
                        sharpenedSR=cv2.medianBlur(sharpenedSR,5)
                    else:
                        sharpenedSR=searchRange.copy()
                    method = eval('cv2.TM_CCOEFF_NORMED')
                    #搜索区域,目标对象
                    cv2.imwrite('/usr/lib/python2.7/site-packages/ori.png',sharpenedSR)
                    cv2.imwrite('/usr/lib/python2.7/site-packages/tar.png',self._targetImg)

                    res = cv2.matchTemplate(sharpenedSR,self._targetImg,method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    #top_left = max_loc
                    #bottom_right = (top_left[0] + w1, top_left[1] + h1)
                    #print('&& lrtb &&')
                    #print(middlePoint)
                    #print(x_left)
                    #print(x_right)
                    #print(y_top)
                    #print(y_bottom)
                    #print(max_val)
                    #print('&&&&')
                    if max_val > self._ratio:
                        #大于相似度，返回中心点以及各参数
                        return middlePoint,status,max_val
                    else:
                        #小于相似度
                        #print('in fall max_val')
                        return middlePoint,status,max_val
                else:
                    #没有中心点，则只能检测是否存在（比较弱的检测，准确度需要控制，通过特征点的多少），无法进行点击
                    #print('in fall no_middle')
                    #print(middlePoint)
                    #print(status)
                    return [None,None],status,None
            else:
                #print('in fall no_Orignal')
                return [None,None],None,None
        else:
            #特征点数不够
            H, status = None, None
            #if we cannot get the middle point, so long as there is one matched point,return it as middle point
            #print('in fall tezheng < 4')
            return [None,None],None,None

    def _filter_matches(self, kp1, kp2, matches):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self._ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, list(kp_pairs)

    def _explore(self, kp_pairs, status = None, H = None):
        h1, w1 = self._targetImg.shape[:2]
        h2, w2 = self._regionImg.shape[:2]
        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        vis[:h1, :w1] =self. _targetImg
        vis[:h2, w1:w1+w2] = self._regionImg
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1],[w1/2,h1/2]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
            #cv2.polylines(vis, [corners], True, (255, 255, 255))
            return (corners[-1])
        else:
            return [None,None]

class region:
    __x=0;
    __y=0;
    __width=0;
    __height=0;

    def __init__(self, x=0, y=0, width=0, height=0):
        self.__x=x;
        self.__y=y;
        self.__width=width;
        self.__height=height;

    def setX(self, x):
        self.__x=x;

    def x(self):
        return self.__x;

    def setY(self, y):
        self.__y=y;

    def y(self):
        return self.__y;

    def setWidth(self, width):
        self.__width=width;

    def width(self):
        return self.__width;

    def setHeight(self, height):
        self.__height=height;

    def height(self):
        return self.__height;
