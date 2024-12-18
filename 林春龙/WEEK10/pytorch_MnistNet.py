# -*- codeing = utf-8 -*-
# @Time : 2024/12/3 22:41
# @File : pytorch_ministnet.py
# @SoftWare : PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                # 每次处理新的一批数据进行反向传播计算梯度之前,将模型参数对应的梯度清零
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                # 计算损失值
                loss = self.cost(outputs, labels)
                # 反向传播基于链式求导法则,计算loss关于模型中所有可学习参数的梯度
                loss.backward()
                # 对于随机梯度下降（SGD）优化器，它会按照 参数 = 参数 - 学习率 * 梯度
                self.optimizer.step()

                # 获取 loss 这个张量对应的标量值
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# 对输入数据的处理变换
def mnist_load_data():
    # 转换为 torch.Tensor 类型
    # 转换到 [0.0, 1.0]之间，以均值为 0、标准差为 1 对数据进行归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # batch_size=32：设置每个批次包含的数据样本数量为 32，
    # 意味着每次迭代从数据集中取出 32 张手写数字图像及其对应的标签作为一组输入到模型中进行训练，
    # 这样可以在训练过程中更好地利用计算资源，同时避免一次性加载全部数据导致内存占用过大等问题。
    # shuffle=True：设置为 True 表示在每个训练周期（epoch）开始时，会对训练集中的数据进行随机打乱顺序，
    # 这样有助于模型在训练过程中能够以不同的顺序看到数据，避免对数据顺序产生依赖，提高模型的泛化能力。
    # num_workers=2：指定用于数据加载的子进程数量为 2，使用多个子进程可以并行地进行数据读取和预处理等操作，
    # 加快数据加载的速度，提升训练效率（但要注意避免过多的子进程带来额外的开销和潜在的资源竞争等问题
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 三个全连接层，指那些包含可学习参数（权重和偏置）
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # 将torch.Size([32, 1, 28, 28]) 变为 torch.Size([32, 784])
        # -1表示自动推断这个维度的大小
        x = x.view(-1, 28*28)
        # 与权重进行矩阵乘后，使用 ReLU 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 经过 softmax 操作后，输出向量的每个元素都在 0 到 1 之间，并且所有元素之和等于 1
        # dim参数用于指定在哪个维度上进行 softmax 操作
        # 张量的维度是从 0 开始计数的。对于形状为 (batch_size, num_classes)（这里就是 (batch_size, 10)）的张量，
        # dim=1 表示沿着第二个维度（也就是类别维度）进行 softmax 操作。
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':

    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    # for i, data in enumerate(train_loader, 0):
    #     inputs, labels = data
    #     print(i)
    #     print(inputs.shape)
    #     x = inputs.view(-1, 28 * 28)
    #     print(x.shape)
    #     print(labels)
    #     break

    model.train(train_loader)
    model.evaluate(test_loader)
