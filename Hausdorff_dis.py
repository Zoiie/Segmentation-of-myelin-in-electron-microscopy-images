import cv2
import matplotlib.pyplot as plt


def get_contours(img):
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]


def main():
    # 1.导入图片
    img_ori = cv2.imread("data/compare/original.png")
    img_base = cv2.imread("data/compare/pred_base.png")
    img_topo = cv2.imread("data/compare/pred_topo.png")

    # 2.获取图片连通域
    cnt_ori = get_contours(img_ori)
    cnt_base = get_contours(img_base)
    cnt_topo = get_contours(img_topo)

    # 3.创建计算距离对象
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()

    # 4.计算轮廓之间的距离
    d1 = hausdorff_sd.computeDistance(cnt_ori, cnt_ori)

    d2 = hausdorff_sd.computeDistance(cnt_ori, cnt_base)

    d3 = hausdorff_sd.computeDistance(cnt_ori, cnt_topo)

    # 5.显示图片
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
    plt.title(u'original.png')
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
    plt.ylabel(u'dis_ori.png')
    plt.xlabel(u'd = ' + '%.4f' % d1)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB))
    plt.ylabel(u'dis_base.png')
    plt.xlabel(u'd = ' + '%.4f' % d2)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img_topo, cv2.COLOR_BGR2RGB))
    plt.ylabel(u'dis_topo.png')
    plt.xlabel(u'd = ' + '%.4f' % d3)
    plt.show()


if __name__ == '__main__':
    main()

