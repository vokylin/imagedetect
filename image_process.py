# coding=utf-8
import opencv_util as ou
import numpy as np
import time
from appium.webdriver.common.touch_action import TouchAction
ratio = 0.75
rate = 0.15
def region(x, y, width, height):#像素点区域
    return ou.region(x, y, width, height)

def regionP(driver,x, y, width, height):#百分比区域
    sWidth=driver.get_window_size()['width'];
    sHeight=driver.get_window_size()['height'];
    return ou.region(x*sWidth, y*sHeight, width*sWidth, height*sHeight)

def tap_element_by_image(driver, img,reference ,original=None, region=None):
    tapPoint=_get_element_middle_point(driver, img, original,region)
    status = tapPoint[2]
    max_val = tapPoint[3]#max_val 模板匹配相似值
    if(tapPoint[0] is not None and tapPoint[1] is not None):
        if(max_val<ratio):#检查相似度，模板匹配小于75，则进行特征点匹配，存在中心点，可以点击
            target = 1-float(abs(reference[0]-np.sum(status)))/float(reference[0])
            base = 1-float(abs(reference[1]-len(status)))/float(reference[1])
            if(target >= rate and base >= rate):#检查特征点是否能大于相似度
                TouchAction(driver).press(x=int(tapPoint[0]),y=int(tapPoint[1])).release().perform()
                return True
            else:
                return False
        else:
            TouchAction(driver).press(x=int(tapPoint[0]),y=int(tapPoint[1])).release().perform()
            return True
    else:
        #如果没有中心点。无论如何不允许点击
        return False;

def is_element_present_by_image(driver, img,reference ,original=None, region=None):
    tapPoint=_get_element_middle_point(driver, img,original, region)
    status = tapPoint[2]
    max_val = tapPoint[3]
    if max_val and np.sum(status) and len(status):
        #print('max_val:%f'%(max_val))
        #print(np.sum(status))
        #print(len(status))
    if(tapPoint[0] is not None and tapPoint[1] is not None):
        if(max_val<ratio):
            target = 1-float(abs(reference[0]-np.sum(status)))/float(reference[0])
            base = 1-float(abs(reference[1]-len(status)))/float(reference[1])
            if(target >= rate and base >= rate):
                return True
            else:
                return False
        else:
            return True
    else:
        if status is not None:
            #存在没有中心点，但是依旧可以用特征点的相似度来确定是否存在匹配
            target = 1-float(abs(reference[0]-np.sum(status)))/float(reference[0])
            base = 1-float(abs(reference[1]-len(status)))/float(reference[1])
            if(target >= rate and base >= rate):
                return True
            else:
                return False
        else:
            return False;

def get_feature_number_by_image(img1,img2, original=None, region=None):
    findElementObj=ou.FindObj(img1, img2,original, region)
    return (findElementObj.findFeatureNumByBrisk())

def _get_element_middle_point(driver, img, original, region):
    i=img.find('.')
    j=img.rfind('/')
    if(j>=0):#picture path is not null
        screenShotFileName=img[0:j+1]+str(time.time())+'-'+img[j+1:i]+'.png'
    else:#picture path is null
        screenShotFileName=str(time.time())+'-'+img[0:i]+'.png'
    driver.get_screenshot_as_file(screenShotFileName)
    #print('**')
    #print(img)
    #print(screenShotFileName)
    #print(original)
    #print('**')
    findElementObj=ou.FindObj(img, screenShotFileName, original, region)
    tapPoint=findElementObj.findMiddlePointByBrisk()
    return tapPoint
