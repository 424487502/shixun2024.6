import cv2

# 处理图像，进行边缘检测
def process_image(image_path):
    original = cv2.imread(image_path)  # 读取原始图像
    processed = cv2.Canny(original, 40, 150)  # 使用Canny算法进行边缘检测
    return original, processed  # 返回原始图像和处理后的图像

# 捕获视频流
def capture_video():
    cap = cv2.VideoCapture(0)  # 打开摄像头
    while True:
        ret, frame = cap.read()  # 读取一帧图像
        im = cv2.flip(frame, 1)  # 水平翻转图像
        if not ret:
            break  # 如果读取失败，退出循环
        ret, jpeg = cv2.imencode('.jpg', im)  # 编码图像为JPEG格式
        if not ret:
            break  # 如果编码失败，退出循环
        frame = jpeg.tobytes()  # 将编码后的图像转换为字节流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # 生成HTTP响应流
    cap.release()  # 释放摄像头

# 捕获视频流并进行边缘检测
def capture_edge_detection():
    cap = cv2.VideoCapture(0)  # 打开摄像头
    while True:
        ret, frame = cap.read()  # 读取一帧图像
        im = cv2.flip(frame, 1)  # 水平翻转图像
        if not ret:
            break  # 如果读取失败，退出循环
        edge_frame = cv2.Canny(im, 40, 150)  # 使用Canny算法进行边缘检测
        ret, jpeg = cv2.imencode('.jpg', edge_frame)  # 编码边缘检测后的图像为JPEG格式
        if not ret:
            break  # 如果编码失败，退出循环
        frame = jpeg.tobytes()  # 将编码后的图像转换为字节流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # 生成HTTP响应流
    cap.release()  # 释放摄像头