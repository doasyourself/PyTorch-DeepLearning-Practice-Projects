import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('./data/5678.png', cv2.IMREAD_GRAYSCALE) # 加载灰度图像

# 二值化
"""
黑色0， 白色255
127是阈值
255是大于阈值时设置的像素值
cv2.THRESH_BINARY_INV是指反转二值化（黑色为前景，白色为背景）
如果用cv2.THRESH_BINARY，则会得到常规的白底黑字二值图像
"""
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize=(10, 5))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示二值化后的图像
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# 展示图像
plt.tight_layout()
plt.show()

#### 轮廓检测
"""
cv2.RETR_EXTERNAL: 表示只检测外部轮廓，不考虑内部轮廓
cv2.CHAIN_APPROX_SIMPLE：使用简单的链式近似法来表示轮廓。它将多余的点压缩成直线段，只八六轮廓的端点，从而减少计算量
"""
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(type(contours))

# 按轮廓的中心点的x坐标排序
def sort_contours(contours):
    # 将轮廓转换为列表
    contours_list = list(contours)
    # 按x坐标排序
    contours_list.sort(key=lambda c: cv2.boundingRect(c)[0])
    return contours_list

# 对轮廓进行排序
contours = sort_contours(contours)

# 遍历轮廓，提取每个数字
digit_images = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h > 20 and w > 10: # 筛选掉过小的区域
        # digit = binary_image[y:y+h, x:x+w]
        padding = 10 # 增加边缘填充
        digit = binary_image[max(y - padding, 0):y+h+padding, max(x - padding, 0):x+w+padding]
        digit_images.append(digit)

len(digit_images)

#### 展示分割之后的图像
plt.figure()
for i in range(len(digit_images)):
    plt.subplot(1, len(digit_images), i+1)
    plt.tight_layout()
    plt.imshow(digit_images[i], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
plt.show()

#### 数字识别
import torch
from PIL import Image
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # (n, 1, 28, 28)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3), # (n, 32, 26, 26)
            torch.nn.BatchNorm2d(32), # (n, 32, 26, 26)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # (n, 32, 13, 13)
        )

        # (n, 32, 13, 13)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3), # (n, 64, 11, 11)
            torch.nn.BatchNorm2d(64), # (n, 64, 13, 13)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # (n, 64, 5, 5)
        )

        # (n, 64, 5, 5)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1600, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(50, 10), # (n, 10)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x 
    
model = Net().to(device)

model_path = '../models/1_Handwritten_Digit_Recognition/model_weights.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
    
#### 预测函数
def predict_image(image, model):
    image = Image.fromarray(image)
    # 图像预处理
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
        max_value, max_index = torch.max(output.data, 1)
    return str(max_index.item())


# 预测
predict_digit = []

for image in digit_images:
    predict_digit.append(predict_image(image, model))

print(''.join(predict_digit))