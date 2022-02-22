import numpy as np
import cv2 as cv
#每个像素都由一个八位整数来表示,即每个像素的值范围是 0~255
# img = np.zeros((3,3),dtype=np.uint8) #创建一个黑色像素
# print(img)
# img=cv.cvtColor(img,cv.COLOR_BAYER_BG2BGR)#把黑色像素转为BGR(Blue-Green-Red)格式
# print(img)
# #输出结果可以看出:每个BGR像素都由一个三元数组表示没并且每个整形向量分别表示一个B,G,R通道
# print(img.shape)#查看图像结构
# img=cv.imread('timg.jpg')#读入文件
# cv.imwrite('filename.jpg',img)#写出文件

gray_img = cv.resize(cv.imread('face_008.jpg', cv.IMREAD_GRAYSCALE), (24, 24))  # 转化大小为24*24的灰度图
cv.imwrite('greyAbc.jpg',gray_img)#写出灰度图片

'''
 注意
 1.即使图像为灰度格式 也会读入BGR格式 BGR 与RGB表示颜色色彩空间相同,但字节顺序相反
 2.无论采用哪种模式读入,imread()函数会删除所有alpha的通道信息(透明度)
 3.imwrite()函数要求图像为BGR或者灰度格式,并且每个通道要有一定的位(bit)输出格式要支持这些通道
    例如,bmp格式要求每个通道又8位,而PNG格式允许每个通道有8位或者16位
'''