# 修复 OpenMP 库冲突问题（Windows 上常见）
# 必须在导入 torch 之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch 
from torch.nn import BatchNorm2d, Linear, MaxPool2d
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import time

### 1.初始化环境
# 隐藏警告
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体保证中文正常显示
plt.rcParams['axes.unicode_minus'] = False # 保证正常显示负号
plt.rcParams['figure.dpi'] = 100 # 设置分辨率，也可以不设置

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device # 输出设备信息

### 下载minist数据集并做预变换处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载训练数据，如果文件存在则不下载
print('加载训练数据...')
train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, download=True, transform=transform)
print('训练数据集长度:', len(train_dataset))

print('加载测试数据...')
test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, download=True, transform=transform)
print('测试数据集长度:', len(test_dataset))

print('数据加载完成')

### 展示MNIST数据集
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.data[i], cmap='gray', interpolation='none')
    print(f"图像尺寸：{train_dataset.data[i].shape}，图像类型：{type(train_dataset.data[i])}")
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

### 构建简单的CNN网络
### 初始输入张量：（batch, 1, 28, 28)，即使b c w h 张量
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__() # 调用父类构造

        ## 初始化网络结构
        # 特征卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3), # 特征层/Conv层，输出(batch, 32, 26, 26)
            torch.nn.BatchNorm2d(32), # 标准化层/BN层
            torch.nn.ReLU(), # 非线性/激活层
            torch.nn.MaxPool2d(kernel_size=2), # 池化 输出：(Batch, 32, 13, 13)
        )
        # 特征卷积层。和第一层不同，进一步扩大通道数
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3), # 特征层/Conv层，输出(batch, 64, 11, 11)
            torch.nn.BatchNorm2d(64), # 标准化层/BN层
            torch.nn.ReLU(), # 非线性/激活层
            torch.nn.MaxPool2d(kernel_size=2), # 池化 输出：(Batch, 64, 5, 5)
        )
        # 全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1600, 50), # 线性映射 64x5x5=1600，展平在forward函数中
            torch.nn.ReLU(), # 非线性
            torch.nn.Dropout(0.5), # 随机关闭，不影响维度
            torch.nn.Linear(50, 10) # 线性映射
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1) # flatten
        x = self.fc(x)
        return x

# 实例化模型
model = Net().to(device)

# 查看模型结构
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量：{total_params:,}")
    print(f"模型可训练参数数量：{trainable_params:,}")

# 打印模型详细结构
print(model) 

# 计算参数数量
count_parameters(model)

### 训练模型
# 配置训练参数
loss_fn = torch.nn.CrossEntropyLoss() # 交叉熵损失函数，常用在多分类任务中
learn_rate = 0.01 # 学习率

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9) # 最后一个参数是动量系数

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 单批次训练函数
def train(dataloader, model, loss_fn, optimizer):
    total_image_count = len(dataloader.dataset)
    num_batches = len(dataloader)

    train_acc, train_loss = 0, 0 # 初始化训练损失和正确率

    # 下面是一个batch的训练
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 计算预测误差
        pred = model(images) # 网络输出 (batch_size, 10)
        loss = loss_fn(pred, labels) # 计算预测值和真实值之间的差距,返回的是 batch 的平均损失，不是总和

        # 反向传播
        optimizer.zero_grad() # grad属性归0
        loss.backward() # 为反向传播计算梯度
        optimizer.step() # 更新参数

        # 记录正确率和损失率
        train_acc += (pred.argmax(1) == labels).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= total_image_count
    train_loss /= num_batches # loss是批次的平均值，所以这里不用除以total_image_count

    return train_acc, train_loss

def test(dataloader, model, loss_fn):
    total_image_count= len(dataloader.dataset) # 获取图片数量
    num_batches = len(dataloader)

    test_loss, test_acc = 0, 0

    with torch.no_grad(): # 相当于c++中的guard
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # 计算loss
            preds = model(images)
            loss = loss_fn(preds, labels)
            
            # 计算总正确数和损失总和
            test_loss += loss.item()
            test_acc += (preds.argmax(1) == labels).type(torch.float).sum().item()

    test_acc /= total_image_count
    test_loss /= num_batches
    return test_acc, test_loss

### 开始多轮训练
epochs = 10

# 数组用于可视化
train_loss = []
train_acc = []
test_loss = []
test_acc = []

start_time = time.perf_counter()

# 训练一次推理一次，加起来算一轮
for epoch in range(epochs):
    # 进入训练模式
    model.train() 
    epoch_train_acc, epoch_train_loss = train(train_loader, model, loss_fn, optimizer)

    # 进入评估模式，也就是推理
    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_loader, model, loss_fn)
    
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = 'Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}'

    # 打印训练和测试结果
    print(template.format(epoch+1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))

end_time = time.perf_counter()
print(f"训练时间: {end_time - start_time:.2f} 秒")

# 可视化训练和测试结果
epochs_range = range(epochs)

plt.figure(figsize=(12, 3))

plt.title("Training and Test Accuracy")
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')

plt.title('Training and Test Loss')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label="Test Loss")
plt.legend(loc='upper right')

plt.show()


### 保存模型
# 保存路径
save_dir = './models/1_Handwritten_Digit_Recognition'

# 确保目录存在，不存在则创建
import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存模型
start_time = time.perf_counter()
torch.save(model.state_dict, os.path.join(save_dir, 'model_weights.pth'))
end_time = time.perf_counter()
print(f"保存模型时间: {end_time - start_time:.2f} 秒")
