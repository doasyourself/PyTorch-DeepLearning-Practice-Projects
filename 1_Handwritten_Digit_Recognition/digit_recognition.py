# 修复 OpenMP 库冲突问题（Windows 上常见）
# 必须在导入 torch 之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch 
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np

# 展示图片
import matplotlib.pyplot as plt

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# 设计神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # (batch, 1, 28, 28)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3), # (batch, 32, 26, 26)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # (batch, 32, 13, 13)
        )

        # (batch, 32, 13, 13)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3), # (batch, 64, 11, 11)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # (batch, 64, 5, 5)
        )
        # 全连接层
        self.fc =  torch.nn.Sequential(
            torch.nn.Linear(1600, 50), # 1600 == 64*5*5
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(50, 10), # 输出
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
# 实例化并加载模型
model = Net().to(device)
model_path = './models/1_Handwritten_Digit_Recognition/model_weights.pth'
model.load_state_dict(torch.load(model_path))

# 将模型设置为评估模式
model.eval()

# 预测函数
def predict_image(image_path, model):
    image = Image.open(image_path)
    
    # 图像预处理（必须与训练时一致！）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        print(output)
        max_value, predicted = torch.max(output.data, 1)
        print(f'Max value: {max_value.item()}, Predicted: {predicted.item()}')
    return predicted.item()

image_path = '1_Handwritten_Digit_Recognition/data/8_1.png'
img = Image.open(image_path)
plt.imshow(img)  # 指定灰度colormap以正确显示灰度图
plt.axis('off')  # 可选，关闭坐标轴
plt.show()

# 使用模型进行预测
predicted_digit = predict_image(image_path, model)
print(f'Predicted digit: {predicted_digit}')
