import numpy as np
import scipy
from PIL import Image
import os
import matplotlib.pylab as plt
import cv2 as cv

def transfer(infile):
    im = Image.open(infile)
    # infile.astype(np.int32)

    # img=cv.imread(infile,0)
    # print(infile)
    im_arr = np.array(im)
    # print("????", im_arr)
    im = Image.fromarray(im_arr)
    print(np.max(im))

    # size = (1300, 1300)
    # print(infile)
    # img = cv.resize(img, size)

    # a = np.asarray(infile)
    # infile = Image.fromarray(a)
    # new_high=1300
    # new_wide=1300
    # reim=infile.resize((new_high, new_wide),Image.ANTIALIAS)
    return im

def split(address,filename,edge):
    # img = Image.open(address)
    #print(size)
    # filename=np.split(address,"/")
    img = address
    img=transfer(img)
    # img = Image.open(address)
    # img.astype(np.int32)
    # print(np.shape(img))
    (size,wide) = np.shape(img)
    # print(img.max(),img.min())

    # 准备将图片切割成小图片
    # weight = int(wide // edge)
    # height = int(size // edge)
    weight = int(65)
    height = int(65)
    # 切割后的小图的宽度和高度
    #print(weight, height)

    for j in range(edge):
        for i in range(edge):
            box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
            region = img.crop(box)

            # region=Image.fromarray(np.array(region))
            # region = region.convert("RGB")
            # region = region.resize((weight, height))
            # plt.imshow(region)
            # plt.show()
            # cropImg = img[weight * i:weight * (i + 1), height * j:height * (j + 1)]
            print(np.array(region))
            if ".png" in filename:
                name=filename.replace(".png","")
                print("}}",region)
                # cv.imwrite('./data/TRAIN/N{}_{}{}.png'.format(i,j, name),region)
                # region = region.resize((weight, height))
                # region.show()
                region.save('./data/TRAIN/N{}_{}{}.png'.format(i,j, name),"png")
                # plt.save('./data/SEGMENT_TRAIN/N{}_{}{}.png'.format(i,j, name),region)

                # plt.imshow(region)
                # plt.show()

            if ".tif" in filename:
                name=filename.replace(".tif","")
                # region = region.resize((weight, height))
                # cv.imwrite('./data/TRAIN/N{}_{}{}.tif'.format(i,j, name),region)
                region.save('./data/TRAIN/N{}_{}{}.tif'.format(i, j, name))

if __name__=="__main__":
    address = r'./data2/TRAIN'
    list=os.listdir(address)
    for i in list:
        filename=address+"/"+i
        # print(i)
        # i=transfer(filename)
        split(filename,i,4)
    # print(list)
    # split(address,2)

# img=Image.open("./Figure_1.png")
# new_img=transfer(img)
# print(np.shape(new_img))
# plt.imshow(new_img)
# plt.show()