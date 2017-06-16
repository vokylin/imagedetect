# coding=utf-8
#This is the image detecting functions file

import numpy as np
import cv2

class FindObj:
    _targetImg = ""#要找的目标图片
    _regionImg = ""#被搜索的大图
    _originalWidth=None#截图时，原始大图的宽度
    _region=None
    _ratio=0.75

    def __init__(self, targetImg, regionImg, originalWidth,region):
        self._region=region
        if self._region is None:
            self._targetImg = cv2.imread(targetImg, 0)
            self._regionImg = cv2.imread(regionImg, 0)
        else:
            self._targetImg = cv2.imread(targetImg, 0)
            regionTempImg = cv2.imread(regionImg, 0)
            tempH, tempW=regionTempImg.shape[:2]
            if(not self._region.height()):
                self._region.setHeight(tempH)
            if(not self._region.width()):
                self._region.setWidth(tempW)
            print('jiequ hou de zuobiao')
            print(self._region.y(),(self._region.height()+self._region.y()), self._region.x(),(self._region.width()+self._region.x()))
            self._regionImg=regionTempImg[self._region.y():(self._region.height()+self._region.y()), self._region.x():(self._region.width()+self._region.x())];
#jiequ hou de tupian            cv2.imwrite('/usr/lib/python2.7/site-packages/shot.png',self._regionImg)
        self._originalWidth = originalWidth

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
        if(self._region is not None and regionX is not None and regionY is not None):
            return [regionX + self._region.x(), regionY + self._region.y(),status,max_val];
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
            print('in fall wrong target Img or retionImg')
            print(self._targetImg)
            print(self._reginImg)
            return [None,None],None,None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        kp1, desc1 = detector.detectAndCompute(self._targetImg, None)
        kp2, desc2 = detector.detectAndCompute(self._regionImg, None)
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = self._filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:#computing a matrix nees 4 matched points at least
 #           print('this.originNal Widt')
 #           print(self._originalWidth)
            if self._originalWidth is not None:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                middlePoint=self._explore(kp_pairs, status, H)
                h1, w1 = self._targetImg.shape[:2]
                h2, w2 = self._regionImg.shape[:2]
#middlePoint[0]----width   middlePoint[1]----height
                if(middlePoint[0] is not None and middlePoint[1] is not None and middlePoint[0]<w2 and middlePoint[1]<h2):
                #baozheng zhongxin dian zai quyu nei
                    #验证中心点坐标两个宽高的范围内，模板匹配的结果
                    #剪切出模板匹配的目标范围图像searchRange
                    zoom_region=float(self._originalWidth)/float(w2)
                    regionImgZoomed=cv2.resize(self._regionImg,(self._originalWidth,int(h2*zoom_region)),cv2.INTER_CUBIC)
                    print('@@ hw @@&')
                    print(h1)
                    print(w1)
                    print(h2)
                    print(w2)
                    print(zoom_region)
                    print(middlePoint)
                    print('&@@@@')
                    x_left=middlePoint[0]*zoom_region-w1
                    x_right=middlePoint[0]*zoom_region+w1
                    y_top=middlePoint[1]*zoom_region-h1
                    y_bottom=middlePoint[1]*zoom_region+h1
                    if x_left<0:
                        x_left=0
                        x_right=w2
                    if y_top<0:
                        y_top=0
                        y_bottom=h2
                    searchRange=regionImgZoomed[y_top:y_bottom, x_left:x_right];
#                    cv2.imwrite('/usr/lib/python2.7/site-packages/shot.png',self._regionImg)
    
                    if w2<self._originalWidth:
                        #锐化
                        sharpenedSR=np.zeros(searchRange.shape, np.uint8)
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                        sharpenedSR = cv2.filter2D(searchRange, -1, kernel)
                        #中值滤波，降噪
                        sharpenedSR=cv2.medianBlur(sharpenedSR,5)
                    else:
                        sharpenedSR=searchRange.copy()
                    method = eval('cv2.TM_CCOEFF_NORMED')
                    #orgi,targetImg
                    cv2.imwrite('/usr/lib/python2.7/site-packages/ori.png',sharpenedSR)
                    cv2.imwrite('/usr/lib/python2.7/site-packages/tar.png',self._targetImg)

                    res = cv2.matchTemplate(sharpenedSR,self._targetImg,method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
                    #top_left = max_loc
                    #bottom_right = (top_left[0] + w1, top_left[1] + h1)
                    print('&& lrtb &&')
                    print(middlePoint)
                    print(x_left)
                    print(x_right)
                    print(y_top)
                    print(y_bottom)
                    print(max_val)
                    print('&&&&')
                    if max_val > self._ratio:
                        return middlePoint,status,max_val
                    else:
                        print('in fall max_val')
                        return middlePoint,status,max_val
                else:
                    print('in fall no_middle')
                    print(middlePoint)
                    print(status)
                    return [None,None],status,None
            else:
                print('in fall no_Orignal')
                return [None,None],None,None
        else:
            H, status = None, None
            #if we cannot get the middle point, so long as there is one matched point,return it as middle point
            print('in fall tezheng < 4')
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
