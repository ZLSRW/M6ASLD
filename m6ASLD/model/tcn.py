import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from pykalman import KalmanFilter
import numpy as np
import math
#即在输入三条序列之后，根据Kalman滤波的方式去调整观测值，观测值实际为TCN的卷积输出。


class PositionalEncoding(nn.Module): #262x1x256
    def __init__(self, d_model=256, dropout=0.1, max_len=2614):  # 10000，默认词长最长为10000
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) #500x256
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 500x1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(500.0) / d_model))  # [128]
        pe[:, 0::2] = torch.sin(position * div_term)  # 间隔赋值
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # print('self.pe.shape '+str(self.pe.shape)) #500x1x256
        # print('xx.shape '+str(x.shape)) #262x1x256
        x = x + self.pe[:x.size(0), :] # 直接在原序列嵌入的基础上加上位置编码
        # print('x(positionE).shape '+str(x.shape))
        return self.dropout(x)  #输出需要为262x1x256
        # return x  #输出需要为262x1x256

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() #即为了保证因果卷积，去除了右侧填充的内容

class TemporalBlock(nn.Module): #时序块
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)) #权重归一化
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout) #一维卷积1

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout) #一维卷积2

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  #下采样（防止维度不一样的情况）
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # print('out.shape '+str(out.shape)) #out.shape torch.Size([16, 600, 80])
        res = x if self.downsample is None else self.downsample(x)
        # print('self.relu(out + res).shape '+ str(self.relu(out + res).shape)) #torch.Size([16, 600, 80])
        return self.relu(out + res)  #这部分张量相加的含义：残差连接，它允许网络以跨层方式传输信息

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = 3
        self.positionencodeing=PositionalEncoding()
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs
            out_channels = num_outputs
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc=nn.Linear(1,out_channels)

    def forward(self, x): #262x1x256
        x=self.positionencodeing(x) #输出需要为262x256x1
        x=x.permute(0,2,1)
        # print(x.shape) #torch.Size([145, 256, 1])
        return x

        # return self.fc(self.network(x))
