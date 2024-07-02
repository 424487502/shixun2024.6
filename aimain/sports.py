from django.shortcuts import render
from django.http import JsonResponse
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

# 定义卷积神经网络结构
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 第一个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二个卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 第三个卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.dropout = nn.Dropout(p=0.5)  # Dropout层
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # 第一个全连接层
        self.fc2 = nn.Linear(512, num_classes)  # 第二个全连接层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv3(x)))  # 卷积 -> ReLU -> 池化
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))  # 全连接层 -> ReLU
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # 最后一个全连接层
        return x  # 返回结果

# 定义类别字典，用于将预测的类别索引转换为类别名称
dic = {
    0: '羽毛球',
    1: '棒球',
    2: '篮球',
    3: '跳水',
    4: '击剑',
    5: '马拉松',
    6: '乒乓球',
    7: '足球',
    8: '游泳',
    9: '排球',
    10: '举重',
    11: '摔跤',
}

# 加载模型
num_classes = len(dic)  # 获取类别总数
model = CNN(num_classes)  # 实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU（如果可用）或CPU
model = model.to(device)  # 将模型移动到设备上

# 加载模型参数
model.load_state_dict(torch.load('./model/sports_cnn.pth', map_location=device))  # 加载模型参数
model.eval()  # 设置模型为评估模式

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

def classify_image(image):
    image = transform(image).unsqueeze(0).to(device)  # 预处理图像并添加批次维度，移动到设备上
    with torch.no_grad():  # 不计算梯度
        outputs = model(image)  # 获取模型输出
        _, predlabel = torch.max(outputs, dim=1)  # 获取预测的类别索引
    return dic[predlabel.item()]  # 返回预测的类别名称