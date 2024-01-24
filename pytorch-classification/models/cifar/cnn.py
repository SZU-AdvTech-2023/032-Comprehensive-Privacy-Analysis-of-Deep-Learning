import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):

  def __init__(self, Max, Min):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

    self.layer3 = nn.Sequential(
      nn.Linear(64 * 8 * 8, 128),
      nn.ReLU(inplace=True)
    )

    self.layer4 = nn.Sequential(
      nn.Linear(128, 100)
    )

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.Max = Max
    self.Min = Min

  def unscale(self,tensor,max_,min_):
    max_ = max_.reshape(1,-1,1,1)
    min_ = min_.reshape(1,-1,1,1)
    return tensor*(max_-min_)+min_

  def forward(self, x):
    x = self.unscale(x,self.Max,self.Min)
    x = self.layer1(x)
    x = self.pool(x)
    x = self.layer2(x)
    x = self.pool(x)
    x = x.view(-1, 64 * 8 * 8)
    x = self.layer3(x)
    x = self.layer4(x)
    return x
  
  def intermediate_forward(self, x, layer_index):
    x = self.unscale(x,self.Max,self.Min)
    x1 = self.layer1(x)
    if layer_index == 0:
      return x1
    
    x2 = self.pool(x1)
    x2 = self.layer2(x2)
    if layer_index == 1:
      return x2
    
    x3 = self.pool(x2)
    x3 = x3.view(-1, 64 * 8 * 8)
    x3 = self.layer3(x3)
    if layer_index == 2:
      return x3
    
    x4 = self.layer4(x3)
    if layer_index == 3:
      return x4
  
  def list_forward(self, x):
    x = self.unscale(x,self.Max,self.Min)
    x1 = self.layer1(x)
    
    x2 = self.pool(x1)
    x2 = self.layer2(x2)
    
    x3 = self.pool(x2)
    x3 = x3.view(-1, 64 * 8 * 8)
    x3 = self.layer3(x3)
    
    x4 = self.layer4(x3)

    return [x1, x2, x3, x4]

if __name__ == '__main__':
  # 数据加载和预处理
  transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
  train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
  test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  images1 = next(iter(train_loader))[0]  # 训练集图片
  labels1 = next(iter(train_loader))[1]  # 训练集标签
  images0 = next(iter(test_loader))[0] # 测试集图片
  labels0 = next(iter(test_loader))[1] # 测试集标签
  
  max1, _ = torch.max(images1.transpose(0, 1).reshape(3, -1), 1)
  min1, _ = torch.min(images1.transpose(0, 1).reshape(3, -1), 1)

  max0, _ = torch.max(images0.transpose(0, 1).reshape(3, -1), 1)
  min0, _ = torch.min(images0.transpose(0, 1).reshape(3, -1), 1)

  Max = torch.max(max0, max1)
  Min = torch.min(min0, min1)

  # 打印训练集和测试集的数据大小
  print(f"Training set size: {len(train_loader.dataset)}")
  print(f"Testing set size: {len(test_loader.dataset)}")

  # 初始化模型、损失函数和优化器
  model = CNN(Max=Max, Min=Min)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

  # 训练模型
  epochs = 50
  for epoch in range(epochs):
      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      # 在每个epoch结束后输出损失
      print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

  # 在测试集上评估模型
  model.eval()

  train_correct = 0
  train_total = 0
  with torch.no_grad():
      for inputs, labels in train_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          train_total += labels.size(0)
          train_correct += (predicted == labels).sum().item()

  train_accuracy = train_correct / train_total
  print(f'Train Accuracy: {train_accuracy * 100:.2f}%')



  test_correct = 0
  test_total = 0
  with torch.no_grad():
      for inputs, labels in test_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          test_total += labels.size(0)
          test_correct += (predicted == labels).sum().item()

  test_accuracy = test_correct / test_total
  print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

  model_path = r'D:\PyProject\SIA\trained_models\CIFAR100\cnn_best_model.pth'
  torch.save(model.state_dict(), model_path)

  print('saved model sucessed!')