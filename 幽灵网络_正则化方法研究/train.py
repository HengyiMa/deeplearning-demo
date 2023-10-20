import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 定义模型
from g_ghost_regnet import g_ghost_regnetx_040
##################
data_folder = os.path.join(os.path.expanduser("~"), "Desktop", "cifar-10-batches-py")
def load_cifar_10_batch(file):
    with open(os.path.join(data_folder, file), 'rb') as fo:
        cifar_data = pickle.load(fo, encoding='bytes')
    return cifar_data
data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
cifar_data_dict = {}
for file in data_files:
    cifar_data_dict.update(load_cifar_10_batch(file))
images = cifar_data_dict[b'data']
labels = cifar_data_dict[b'labels']

# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

images = images.reshape((-1, 3, 32, 32))  # 将图像数据重塑为3x32x32
images = images.transpose((0, 2, 3, 1))  # 调整图像通道顺序
images = [transform(image) for image in images]

# 数据划分为训练集和验证集
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(torch.stack(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(torch.stack(X_val), torch.tensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = TensorDataset(torch.stack(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = g_ghost_regnetx_040()  

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
num_epochs = 20

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print every 25 batches
        if (i + 1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
    
    train_accuracy = 100 * correct / total
    train_losses.append(total_train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    
    # 在验证集上评估模型
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = total_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    val_accuracies.append(accuracy)

print("Training completed!")
# 在测试集上评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'cifar10_g_ghost_regnetx.pth')

# 绘制训练和验证损失以及训练集准确率
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()