import cv2
import os
import numpy as np
path = './dice/'
files = os.listdir(path)
s2 = cv2.imread("./guanfangyanmo/022.png", 0)# 模板
row, col = s2.shape[0], s2.shape[1]
d = []
for file in files:
    s1 = cv2.imread(path+file, 0)# 读取配准后图像
    print(file)
    s = []
    for r in range(row - 10):
        for c in range(col - 10):
            if s1[r][c] == s2[r][c]:# 计算图像像素交集
                s.append(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
    d.append(2*m1/m2)
#     print(2*m1/m2)
print(d)
