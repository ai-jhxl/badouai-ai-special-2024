# -*- codeing = utf-8 -*-
# @Time : 2024/12/3 20:29
# @File : BP_NeuralNetWork.py
# @SoftWare : PyCharm

import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hidennodes, outputnodes, learngrate):
        # 输入层节点；中间隐藏层节点；输出层节点；学习率
        self.inodes = inputnodes
        self.hnodes = hidennodes
        self.onodes = outputnodes
        self.lr = learngrate

        # 初始化输入层到隐藏层的权重
        # 0为均值, 根号隐藏层节点的平方的倒数为方差，隐藏层节点数为行数，输入节点数为列数的矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 初始化隐藏层到输出层的权重
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 设置 sigmoid 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        # 输入层节点数为行数，1列的二维矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 输出层节点数为行数，1列的二维矩阵
        targets = np.array(target_list, ndmin=2).T

        # 矩阵乘,计算输入层经过权重后的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs

        # 根据梯度下降和链式法则，求解链路权重的更新量
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))

        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(inputs))

    def test(self, test_list):
        # 根据测试数据计算输出,一次正向传播
        hidden_inputs = np.dot(self.wih, test_list)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 初始化网络
# 输入图片总共有28*28 = 784个数值
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 加入epoch,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    # 把数据依靠','区分，并分别读入,第一个为该图片数据的数字号码
    for record in training_data_list:
        all_values = record.split(',')
        # 输入数据标准化Normalization,数据统一映射到[[0.01,0.99]之间
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 测试数据，有10行数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.test(inputs)
    # 找到数值最大的神经元对应的编号，返回给定数组中最大值的索引
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = np.asarray(scores)
print("performance = ", scores_array.sum() / scores_array.size)
