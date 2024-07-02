import cv2
import numpy as np
import base64
from django.core.files.uploadedfile import InMemoryUploadedFile


def count_coins(image: InMemoryUploadedFile):
    # 将上传的文件转为OpenCV图像
    img_array = np.fromstring(image.read(), np.uint8)
    im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 灰度处理
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, im2 = cv2.threshold(imgray, 130, 255, cv2.THRESH_BINARY_INV)

    # 开运算，分割细小链接线
    k = np.ones((10, 10))
    im3 = cv2.morphologyEx(im2, cv2.MORPH_OPEN, k, iterations=2)

    # 腐蚀，缩小硬币区域
    im4 = cv2.erode(im3, k, iterations=2)

    # 查找外轮廓
    cnts, _ = cv2.findContours(im4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 绘制最小外接圆
    for i in cnts:
        (x, y), r = cv2.minEnclosingCircle(i)
        center = (int(x), int(y))
        r = int(r)
        cv2.circle(im, center, r, (0, 0, 255), 2)

    # 将处理后的图像编码为base64
    _, buffer = cv2.imencode('.jpg', im)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return {
        'count': len(cnts),
        'image_base64': img_str
    }