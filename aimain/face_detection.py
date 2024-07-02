import os
import cv2
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import LabelEncoder
from .models import *

# 生成摄像头视频流
def gen(camera):
    try:
        while True:
            ret, frame = camera.read()  # 读取摄像头帧
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 水平翻转帧
            ret, jpeg = cv2.imencode('.jpg', frame)  # 将帧编码为JPEG格式
            if not ret:
                break
            frame = jpeg.tobytes()  # 将编码后的帧转换为字节流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # 生成HTTP响应流
    finally:
        camera.release()  # 释放摄像头

# 人脸识别登录
@csrf_exempt
def face_recognition_login(request):
    cap = None

    try:
        # 加载人脸检测分类器
        cascade_path = os.path.abspath('./model/haarcascade_frontalface_default.xml')
        if not os.path.isfile(cascade_path):
            return HttpResponse("未找到分类器文件")

        fd = cv2.CascadeClassifier(cascade_path)  # 初始化分类器
        if fd.empty():
            return HttpResponse("加载分类器文件失败")

        # 搜索训练数据
        train_faces = search_faces("./data/face/training")

        codec = LabelEncoder()
        codec.fit(list(train_faces.keys()))
        train_x, train_y = [], []

        # 处理每个标签对应的图像文件
        for label, filenames in train_faces.items():
            for file in filenames:
                image = cv2.imread(file)  # 读取图像文件
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
                faceresults = fd.detectMultiScale(gray, 1.3, 2, minSize=(100, 100))  # 检测人脸
                for x, y, w, h in faceresults:
                    train_x.append(gray[y:y + h, x:x + w])  # 提取人脸区域
                    train_y.append(codec.transform([label])[0])  # 添加标签
        train_y = np.array(train_y)

        if len(train_x) < 2 or len(set(train_y)) < 2:
            return HttpResponse("训练数据不足，无法进行人脸识别")

        # 训练LBPH人脸识别模型
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(train_x, train_y)

        cap = cv2.VideoCapture(0)  # 打开摄像头
        if not cap.isOpened():
            return HttpResponse("无法打开摄像头")

        while True:
            ret, frame = cap.read()  # 读取摄像头帧
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
            faceresults = fd.detectMultiScale(gray, 1.3, 2, minSize=(100, 100))  # 检测人脸

            for x, y, w, h in faceresults:
                test_x = gray[y:y + h, x:x + w]  # 提取人脸区域
                pred_y, sc = model.predict(test_x)  # 预测人脸
                label = codec.inverse_transform([pred_y])  # 反转标签编码
                if sc >= 20:
                    user_name = label[0]  # 获取用户名
                    user = User.objects.filter(uname=user_name).first()  # 查询用户
                    if user:
                        request.session['user'] = {'uname': user_name}  # 设置会话
                        return render(request, 'index.html', {'uname': user_name})
                    else:
                        return HttpResponse("检测到的用户不存在")
                else:
                    return HttpResponse("没有特别符合的人脸信息！")

            if cv2.waitKey(1) == 27:  # 按下ESC键退出
                break

        return HttpResponse("人脸识别失败")

    except cv2.error as e:
        return HttpResponse(f"OpenCV错误: {str(e)}")
    except Exception as e:
        return HttpResponse(f"发生错误: {str(e)}")
    finally:
        if cap is not None:
            cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 销毁所有窗口

# 搜索人脸训练数据
def search_faces(dir):
    dir = os.path.normpath(dir)  # 规范化路径
    if not os.path.isdir(dir):
        print("目标不是文件夹！")
        return {}
    faces = {}
    for curdir, subdir, files in os.walk(dir):  # 遍历目录
        for file in files:
            path = os.path.join(curdir, file)
            label = os.path.basename(os.path.dirname(path))
            if label not in faces:
                faces[label] = []
            faces[label].append(path)
    return faces  # 返回人脸数据字典