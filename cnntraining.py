import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整输入图像大小为128x128
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.RandomRotation(10),  # 随机旋转图像
    transforms.ToTensor(),  # 转换图像为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 按RGB通道减去均值
        std=[0.229, 0.224, 0.225]  # 按RGB通道除以标准差
    )
])

# 准备训练集并进行预处理
train_dataset = datasets.ImageFolder(root='/Users/550c/Desktop/shixun/sports', transform=transform)  # 加载图像数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # 创建数据加载器

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
        x = x.view(x.size(0), -1)  # 展平张量
        x = F.relu(self.fc1(x))  # 全连接层 -> ReLU
        x = self.dropout(x)  # Dropout层
        x = self.fc2(x)  # 最后一个全连接层
        return x  # 返回预测结果

num_classes = 12  # 根据实际的体育运动种类数进行修改
model = CNN(num_classes)  # 实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU（如果可用）或CPU
model = model.to(device)  # 将模型移动到设备上

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义Adam优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 定义学习率调度器

# 训练模型
epochs = 25  # 定义训练的轮数
model.train()  # 设置模型为训练模式

for epoch in range(epochs):
    running_loss = 0.0  # 累积损失
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备上
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item() * inputs.size(0)  # 累加损失
        _, predlabel = torch.max(outputs, 1)  # 获取预测结果
        total += labels.size(0)  # 累加总样本数
        correct += (predlabel == labels).sum().item()  # 累加正确预测的数量

    epoch_loss = running_loss / total  # 计算平均损失
    epoch_acc = 100 * correct / total  # 计算准确率
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")  # 输出当前轮次的损失和准确率

    scheduler.step()  # 更新学习率

# 保存模型
torch.save(model.state_dict(), 'sports_cnn.pth')  # 保存模型参数到文件