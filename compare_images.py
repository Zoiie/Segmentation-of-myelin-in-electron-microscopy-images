# coding=utf-8
# 导入python包
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def mse(imageA, imageB):
    # 计算两张图片的MSE指标
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # 返回结果，该值越小越好
    return err

def compare_images(imageA, imageB, title,dice):
    # 分别计算输入图片的MSE和SSIM指标值的大小
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # 创建figure
    fig = plt.figure(title)
    plt.suptitle("%s:\nMSE: %.2f, SSIM: %.2f, Dice: %.2f" % (title,m, s,dice))

    # 显示第一张图片
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # 显示第二张图片
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def resize(img):
    img=cv2.resize(img,(640,426))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresh

def reversal(img):
    img=255-img
    img=np.rot90(img,3)
    img=cv2.flip(img,1)
    return img

def cal_dice(orig,target):
    # path = './dice/'
    # files = os.listdir(path)
    s2 = orig  # 模板
    row, col = s2.shape[0], s2.shape[1]
    # d = []

    s1 = target # 读取配准后图像
    # print(file)
    s = []
    for r in range(row - 10):
        for c in range(col - 10):
            if s1[r][c] == s2[r][c]:  # 计算图像像素交集
                s.append(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
    # d.append(2 * m1 / m2)
    print(2*m1/m2)
    return 2*m1/m2
    # print(d)

# 读取图片
original1 = cv2.imread("./data/compare/original.png")
original1_resize=reversal(original1)
original1_resize=resize(original1_resize)
pred_base = cv2.imread("./data/compare/pred_base.png")
pred_base_resize=resize(pred_base)
pred_topo = cv2.imread("./data/compare/pred_topo.png")
pred_topo_resize=resize(pred_topo)

# 初始化figure对象
fig = plt.figure("Images")
images = ("Original", original1_resize), ("Pred base", pred_base_resize), ("Pred topo", pred_topo_resize)

# 遍历每张图片
for (i, (name, image)) in enumerate(images):
    # 显示图片
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")
plt.tight_layout()
plt.show()

# 比较图片
dice_orig=cal_dice(original1_resize,original1_resize)
dice_base=cal_dice(original1_resize,pred_base_resize)
dice_topo=cal_dice(original1_resize,pred_topo_resize)

compare_images(original1_resize, original1_resize, "Original vs Original",dice_orig)
compare_images(original1_resize, pred_base_resize, "Original vs Pred base",dice_base)
compare_images(original1_resize, pred_topo_resize, "Original vs pred topo",dice_topo)
