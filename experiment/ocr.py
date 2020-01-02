import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os,glob

def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

def ocr_single_image(img_path):
    # 读取图片
    imagePath = img_path
    img = cv2.imread(imagePath)
    
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    orig = img.copy()
    
    # 调用 MSER 算法
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    # cv2.imshow('img', img)
    img_pil = cv2pil(img)
    # plt.imshow(img)
    # plt.show()
    
    # 将不规则检测框处理成矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
    # cv2.imshow("hulls", vis)
    vis_pil = cv2pil(vis)
    # plt.imshow(vis_pil)
    # plt.show()

    return img_pil,vis_pil

def canny_single_image(img_path):
    img = cv2.imread(img_path, 0)#转化为灰度图
    img_color = img
    img_color_cp = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
    
    # gray = cv2.cvtColor(img_color_cp, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_color_cp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thread_pil = cv2pil(thresh)

    blur_pil = cv2pil(blur)
    canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值
    canny_pil = cv2pil(canny)

    return blur_pil,canny_pil,thread_pil

def sharp_single_image(img_path):
    image = cv2.imread(img_path)
    #自定义卷积核
    kernel_sharpen_1 = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1]])
    kernel_sharpen_2 = np.array([
            [1,1,1],
            [1,-7,1],
            [1,1,1]])
    kernel_sharpen_3 = np.array([
            [-1,-1,-1,-1,-1],
            [-1,2,2,2,-1],
            [-1,2,8,2,-1],
            [-1,2,2,2,-1], 
            [-1,-1,-1,-1,-1]])/8.0
    #卷积
    output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
    output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
    output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
    
    return cv2pil(output_1),cv2pil(output_2),cv2pil(output_3)

def ocr_folder(folder_path):
    for img in glob.glob(folder_path):
        a,b = ocr_single_image(img)
        basename = os.path.basename(img)
        a_path = os.path.join('../ocrs/',"a_{}".format(basename))
        b_path = os.path.join('../ocrs/',"b_{}".format(basename))
        print(a_path)
        print(b_path)
        a.save(a_path)
        b.save(b_path)

def canny_folder(folder_path):
    for img in glob.glob(folder_path):
        a,b,c = canny_single_image(img)
        basename = os.path.basename(img)
        a_path = os.path.join('../cannys/',"a_{}".format(basename))
        b_path = os.path.join('../cannys/',"b_{}".format(basename))
        c_path = os.path.join('../cannys/',"c_{}".format(basename))

        print(a_path)
        print(b_path)
        a.save(a_path)
        b.save(b_path)
        c.save(c_path)

# canny_folder("../inputs2/*")
# a,b,c = sharp_single_image("163.jpg")
# a.save("a_163.jpg")
# b.save("b_163.jpg")
# c.save("c_163.jpg")

ocr_folder("../cannys/*")
