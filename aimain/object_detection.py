from django.http import JsonResponse
from ultralytics import YOLO
import cv2

# 控制视频捕获的全局变量，用于开启或关闭视频检测
is_detecting = False


# 从摄像头逐帧生成视频
def gen_frames():
    # 初始化YOLO模型，加载预训练模型文件
    yolo = YOLO('./model/yolov8s.pt')
    # 打开摄像头
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("无法打开摄像头")
        exit()

    # 检测是否在进行中
    while is_detecting:
        ret, frame = cap1.read()  # 读取摄像头帧
        im = cv2.flip(frame, 1)  # 水平翻转帧
        if not ret:
            print("无法读取帧")
            break
        else:
            results = yolo.predict(im)  # 使用YOLO模型进行预测
            im2 = results[0].plot()  # 绘制预测结果
            ret, buffer = cv2.imencode('.jpg', im2)  # 将帧编码为JPEG格式
            if not ret:
                print("无法编码帧")
                break
            frame = buffer.tobytes()  # 将编码后的帧转换为字节流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 一帧一帧地显示结果

    cap1.release()  # 释放摄像头


# 开始视频检测
def start_detection():
    global is_detecting
    is_detecting = True
    print("检测已开始")
    # return JsonResponse({'status': '检测已开始'})


# 停止视频检测
def stop_detection():
    global is_detecting
    is_detecting = False
    print("检测已停止")
    # return JsonResponse({'status': '检测已停止'})