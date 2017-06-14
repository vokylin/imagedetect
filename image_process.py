# coding=utf-8
import opencv_util as ou
import numpy as np
import time
from appium.webdriver.common.touch_action import TouchAction
ratio = 0.75
rate = 0.2
def region(x, y, width, height):#像素点区域
    return ou.region(x, y, width, height)

def regionP(driver,x, y, width, height):#百分比区域
    sWidth=driver.get_window_size()['width'];
    sHeight=driver.get_window_size()['height'];
    return ou.region(x*sWidth, y*sHeight, width*sWidth, height*sHeight)

def tap_element_by_image(driver, img,reference ,originalWidth=None, region=None):
    tapPoint=_get_element_middle_point(driver, img, originalWidth,region)
    status = tapPoint[2]
    max_val = tapPoint[3]#max_val xiangsizhi
    if(tapPoint[0] is not None and tapPoint[1] is not None):
        if(max_val<ratio):#jinru xiangsi du
            target = 1-float(abs(reference[0]-np.sum(status)))/float(reference[0])
            base = 1-float(abs(reference[1]-len(status)))/float(reference[1])
            if(target >= rate and base >= rate):
                TouchAction(driver).press(x=int(tapPoint[0]),y=int(tapPoint[1])).release().perform()
                return True
            else:
                return False
            return True;
        else:
            return True
    else:
        return False;

def is_element_present_by_image(driver, img,reference ,originalWidth=None, region=None):
    tapPoint=_get_element_middle_point(driver, img,originalWidth, region)
    status = tapPoint[2]
    max_val = tapPoint[3]#max_val xiangsizhi
    print('max_val:%f'%(max_val))
    if(tapPoint[0] is not None and tapPoint[1] is not None):
        if(max_val<ratio):
            target = 1-float(abs(reference[0]-np.sum(status)))/float(reference[0])
            base = 1-float(abs(reference[1]-len(status)))/float(reference[1])
            if(target >= rate and base >= rate):
                return True
            else:
                return False
            return True;
        else:
            return True
    else:
        return False;

def get_feature_number_by_image(img1,img2, originalWidth=None, region=None):
    findElementObj=ou.FindObj(img1, img2,originalWidth, region)
    return (findElementObj.findFeatureNumByBrisk())

def _get_element_middle_point(driver, img, originalWidth, region):
    i=img.find('.')
    j=img.rfind('/')
    if(j>=0):#picture path is not null
        screenShotFileName=img[0:j+1]+str(time.time())+'-'+img[j+1:i]+'.png'
    else:#picture path is null
        screenShotFileName=str(time.time())+'-'+img[0:i]+'.png'
    driver.get_screenshot_as_file(screenShotFileName)
    print('**')
    print(img)
    print(screenShotFileName)
    print(originalWidth)
    print('**')
    findElementObj=ou.FindObj(img, screenShotFileName, originalWidth, region)
    tapPoint=findElementObj.findMiddlePointByBrisk()
    return tapPoint
