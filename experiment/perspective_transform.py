import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./1.jpg') # 1008*756 w*h

rows, cols = img.shape[:2]
# 原图中书本的四个角点(x,y)
pts1 = np.float32([[243,430], [289, 547], [890, 245], [964, 300]])
# 变换后分别在左上、左下、右上，右下四个点
pts2 = np.float32([[0, 400], [0, 560], [800, 400], [800, 560]])
# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(img, M, (800, 800))
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# img[:, :, ::-1]是将BGR转化为RGB
plt.show()