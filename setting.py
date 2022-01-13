import cv2 as cv
import numpy as np
#读取需要进行边缘检测的图片
img=cv.imread("THRESH_MEAN.jpg")
# img=img/np.max(img)
#Trackbar的回调函数，我们这里什么也不做
def nothing(x):
    pass
#创建一个边缘检测Edges窗口
cv.namedWindow("Edges")
#创建Canny中maxVal和minVal数值滑动条
cv.createTrackbar("maxVal","Edges",1,10,nothing)
cv.createTrackbar("minVal","Edges",5,10,nothing)
#创建一个开关，方便我们切换查看原图和边缘检测后的图
switch="0:OFF\n1:ON"
cv.createTrackbar(switch,"Edges",0,1,nothing)
#初始化minVal和maxVal
minVal,maxVal=0,10
edges = cv.Canny(img, minVal, maxVal)

#实时更新图片显示
while(1):
    cv.imshow("Edges", edges)
    k=cv.waitKey(1)
    if k == 27:
        break
    #获取Trackbar数值
    maxVal = cv.getTrackbarPos("maxVal", "Edges")
    minVal = cv.getTrackbarPos("minVal", "Edges")
    #设置开关，0为原图，1为边缘检测后的图片
    s = cv.getTrackbarPos(switch, "Edges")
    if s==0:
        edges =img
    else:
        edges = cv.Canny(img, minVal, maxVal)
#最后不要忘记销毁窗口
cv.destorayWindows()
