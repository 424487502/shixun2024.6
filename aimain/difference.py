import base64
import cv2
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile

def find_differences(image1: InMemoryUploadedFile, image2: InMemoryUploadedFile) -> str:
    # 将上传的文件转换为OpenCV可以处理的格式
    im1 = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_COLOR)
    im2 = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_COLOR)

    # 确保两张图片的尺寸相同
    im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

    # 转换为灰度图
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # 计算两张图片的绝对差值
    diff = cv2.absdiff(im1_gray, im2_gray)

    # 二值化
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 形态学操作（膨胀）以连接差异区域
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    # 查找差异区域的轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在两张原图上绘制轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 合并两张图片
    result_image = np.hstack((im1, im2))

    # 编码为Base64字符串
    _, buffer = cv2.imencode('.png', result_image)
    result_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return result_image_base64