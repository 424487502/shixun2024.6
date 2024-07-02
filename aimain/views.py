import base64
from django import forms
from django.views.decorators import gzip
import os
import numpy as np
import pymysql
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from sklearn.preprocessing import LabelEncoder
from .models import *  # 导入所有实体类
import requests
import re
from urllib.parse import quote
from .object_detection import *
from .edge_detection import *
from .difference import *
from .sports import *
from . import face_detection
import cv2
from .counter import *
from .voice_to_words import transcribe_audio
# Create your views here.

# 测试视图
def func(request):
    return HttpResponse('这是应用aimain中的func响应')

# 人脸识别登录视频流
def video_feed(request):
    # 使用StreamingHttpResponse返回视频流
    return StreamingHttpResponse(face_detection.gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

# 渲染人脸识别页面
def face_detection_page(request):
    return render(request, 'face_detection.html')

# 人脸识别登录处理
def face_recognition_login(request):
    return face_detection.face_recognition_login(request)

# 登录视图
def log(request):
    if request.method == 'GET':
        return render(request, 'login.html')  # 如果是GET请求，渲染登录页面
    else:
        name = request.POST['username']
        pwd = request.POST['password']
        # 检查用户名和密码是否匹配
        if User.objects.filter(uname=name, upwd=pwd):
            request.session['user'] = {'uname': name}  # 将用户名存储到会话中
            return render(request, 'index.html', {'uname': name})  # 登录成功，渲染主页
        else:
            return HttpResponse('<h1 style="color:red">登录失败<h1>')  # 登录失败，返回错误信息

# 注册视图
def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')  # 如果是GET请求，渲染注册页面
    else:
        username = request.POST['username']
        password = request.POST['password']
        password2 = request.POST['password2']
        utele = request.POST.get('utel')

        if password != password2:
            return HttpResponse('<h1 style="color:red">两次密码不一致</h1>')  # 检查两次输入的密码是否一致

        try:
            # 连接到 MySQL 数据库
            connection = pymysql.connect(
                host='localhost',
                user='root',
                password='xiaodu123456',
                database='ai',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )

            with connection.cursor() as cursor:
                # 插入用户数据
                sql = "INSERT INTO userinfos (uname, upwd, utel, isvip, isactive) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (username, password, utele, False, True))
                connection.commit()  # 提交更改

            return render(request, 'register_success.html')  # 注册成功，渲染成功页面
        except pymysql.IntegrityError:
            return HttpResponse('<h1 style="color:red">用户名已存在</h1>')  # 用户名已存在
        except Exception as e:
            return HttpResponse(f'<h1 style="color:red">注册失败: {e}</h1>')  # 注册失败，返回错误信息
        finally:
            connection.close()  # 关闭数据库连接

# 渲染“轻松一下”页面
def dy(request):
    return render(request, 'dy.html')

# 渲染百度页面
def baidu(request):
    return render(request, 'baidu.html')

# 数据采集视图
def sjcj(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        codename = quote(name)  # 将名称进行URL编码
        url = "https://image.so.com/i?q=" + codename + "&inact=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url=url, headers=headers)  # 发送请求获取页面
        html = resp.text

        # 使用正则表达式找到所有图片链接
        results = re.findall('"thumb":"(.*?)"', html)
        images = [img.replace('\\', '') for img in results]  # 返回所有找到的图片链接
        indexed_images = [(idx+1, img) for idx, img in enumerate(images)]  # 添加索引

        return render(request, 'sjcj.html', {'images': indexed_images, 'name': name})  # 渲染页面并传递图片数据
    return render(request, 'sjcj.html')

# 渲染目标检测页面
def Object_Detection_Page(request):
    return render(request, 'Object_Detection_Page.html')

# 目标检测 YOLO 视频流
@gzip.gzip_page
def get_video(request):
    # 返回目标检测的视频流
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace;boundary=frame')

# 开始检测视图
def start_detection_view(request):
    start_detection()  # 调用开始检测的函数
    return JsonResponse({'status': '检测已开始'})  # 返回检测已开始的JSON响应

# 停止检测视图
def stop_detection_view(request):
    stop_detection()  # 调用停止检测的函数
    return JsonResponse({'status': '检测已停止'})  # 返回检测已停止的JSON响应

# 渲染边缘检测页面
def edge_detection_page(request):
    return render(request, 'edge_detection.html')

# 上传图像进行边缘检测
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = f'/tmp/{image.name}'
        # 保存上传的图片到临时目录
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        original, processed = process_image(image_path)  # 调用边缘检测函数处理图像
        _, original_encoded = cv2.imencode('.jpg', original)
        _, processed_encoded = cv2.imencode('.jpg', processed)
        original_base64 = base64.b64encode(original_encoded).decode('utf-8')
        processed_base64 = base64.b64encode(processed_encoded).decode('utf-8')
        # 返回原图和处理后图像的base64编码
        return JsonResponse({
            'original': original_base64,
            'processed': processed_base64
        })
    return JsonResponse({'error': '无效的请求'}, status=400)

# 视频流页面
def stream_camera(request):
    # 返回摄像头的视频流
    return StreamingHttpResponse(capture_video(), content_type='multipart/x-mixed-replace; boundary=frame')

# 边缘检测视频流
def stream_edge(request):
    # 返回边缘检测的视频流
    return StreamingHttpResponse(capture_edge_detection(), content_type='multipart/x-mixed-replace; boundary=frame')

# 找茬视图类
class ImageUploadForm(forms.Form):
    image1 = forms.ImageField()
    image2 = forms.ImageField()

# 找茬视图
def find_differences_view(request):
    result = False
    image1_url, image2_url, result_image_base64 = "", "", ""

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image1 = request.FILES['image1']
            image2 = request.FILES['image2']

            result_image_base64 = find_differences(image1, image2)  # 调用找茬函数处理图像
            result = True

    else:
        form = ImageUploadForm()

    return render(request, 'difference.html', {
        'form': form,
        'result': result,
        'result_image_base64': result_image_base64  # 渲染找茬页面并传递结果图像
    })

# 运动分类视图
def sports_classify_view(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image = Image.open(BytesIO(image.read()))  # 从上传的文件中读取图像
        result = classify_image(image)  # 调用运动分类函数处理图像
        return JsonResponse({'result': result})  # 返回分类结果
    return render(request, 'sports.html')

# 数量检测视图
def counter(request):
    result = None
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        result = count_coins(image)  # 调用数量检测函数处理图像

    return render(request, 'counter.html', {'result': result})  # 渲染数量检测页面并传递结果


# 语音识别
def voice_to_words_view(request):
    if request.method == 'POST' and 'audio_file' in request.FILES:
        audio_file = request.FILES['audio_file']
        transcribed_text = transcribe_audio(audio_file)
        return JsonResponse({'transcribed_text': transcribed_text})
    return render(request, 'voice_to_words.html')