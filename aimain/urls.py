from django.urls import path
from . import views

urlpatterns = [
    path('', views.log, name='log'),  # 主页面路径为登录页面
    path('register/', views.register, name='register'), #注册
    #快速链接
    path('dy/', views.dy), #douyin
    path('baidu/', views.baidu), #baidu
    #数据采集
    path('sjcj/',views.sjcj),
    #人脸识别登录
    path('video_feed/', views.video_feed, name='video_feed'),
    path('face_detection/', views.face_detection_page, name='face_detection'),
    path('face-recognition-login/', views.face_recognition_login, name='face_recognition_login'),
    #目标检测
    path('Object_Detection_Page/', views.Object_Detection_Page, name='Object_Detection_Page'),
    path('get_video/', views.get_video, name='get_video'),
    path('start_detection/', views.start_detection_view, name='start_detection'),
    path('stop_detection/', views.stop_detection_view, name='stop_detection'),
    #边缘检测
    path('edge_detection/', views.edge_detection_page, name='edge_detection'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('stream_camera/', views.stream_camera, name='stream_camera'),
    path('stream_edge/', views.stream_edge, name='stream_edge'),
    #快速找茬
    path('difference/', views.find_differences_view, name='find_differences'),
    #运动检测
    path('sports/', views.sports_classify_view, name='classify'),
    #数量检测
    path('counter/', views.counter, name='counter'),
    #语音识别
    path('voice_to_words/', views.voice_to_words_view, name='voice_to_words'),
]