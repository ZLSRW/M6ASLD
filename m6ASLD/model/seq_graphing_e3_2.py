import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
from .tcn_e3_2 import *
# from .Causal_Dilate_Network import *
import pandas as pd
import csv

import random


fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)  # 计算权重
        self.activation = activation

    def forward(self, adj, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        # print(x.size())
        return outputs


class Conv1D_feature_extracter(nn.Module):  # 时序块
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, dropout):
        super(Conv1D_feature_extracter, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           padding=padding, stride=stride))  # 权重归一化
        # self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)  # 一维卷积1

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    padding=padding, stride=stride))
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout) #一维卷积2

        self.net = nn.Sequential(self.conv1, self.tanh1, self.dropout1)
        # self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
        #                          self.conv2, self.relu2, self.dropout2)
        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  #下采样（防止维度不一样的情况）
        # self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print(x.shape) #torch.Size([706, 256, 41])
        out = self.net(x)
        # print(out.shape)
        # print('out.shape '+str(out.shape)) #out.shape torch.Size([16, 600, 80])
        # res = x if self.downsample is None else self.downsample(x)
        # print('self.relu(out + res).shape '+ str(self.relu(out + res).shape)) #torch.Size([16, 600, 80])
        return self.tanh1(out)


class IMILmask(nn.Module):
    def __init__(self, inputsize, outputsiz):
        super(IMILmask, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsiz
        shape1D = int(self.inputsize / self.outputsize)
        self.L1 = nn.Sequential(
            nn.Linear(shape1D, shape1D),
            # nn.relu()
            # nn.relu()
            nn.ReLU()
            # nn.Tanh()
            # nn.LeakyReLU()
        )

        self.L2 = nn.Sequential(
            nn.Linear(shape1D, shape1D),
            # nn.Tanh()
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(shape1D, shape1D)  # 注意力权重
        return

    def forward(self, x):
        # x = x.permute(0,2,1)
        # print('x_IMIL.shape '+str(x.shape))
        l1 = self.L1(x)  # 32x12x20
        l2 = self.L2(x)  # 32x12x20
        A_imi = self.attention_weights(l1 + l2)  # element wise multiplication(未实验)
        # A_imi=F.softmax(A_imi)
        A_imi = torch.sigmoid(A_imi)
        # A_imi=torch.relu(A_imi)
        x = torch.mul((A_imi + 1), x)
        # print('A_imi.shape '+str(A_imi.shape))
        # print('x.shape '+str(x.shape))
        return A_imi, x


class FeatureConverge(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(FeatureConverge, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.cov1D_extracter1 = Conv1D_feature_extracter(self.inputsize, self.inputsize, 3, 1, 1, 0.2)
        # self.mask = IMILmask(self.inputsize, self.outputsize)
        # self.cov1D_extracter2 = Conv1D_feature_extracter(self.outputsize, self.outputsize, 3, 1, 1, 0.2)  # 通过1维卷积将其转为包状态
        self.cov2D_extracter2 = nn.Conv2d(in_channels=self.outputsize, out_channels=self.outputsize, kernel_size=3,
                                          padding=1)
        # 对输入序列进行执行一维卷积
        # 计算注意力权重
        # 生成两个分支，一个分支为包嵌入的包分类（120x10x1），一个分支为包内结合位点的分类(120x260x1)
        self.to('cuda:0')
        return

    def mapping_converge(self, x, index_all):
        # index_all 226x4x64的列表
        # print(x.shape) #torch.Size([706, 256, 41])
        # print(index_all)
        a, b, c = x.size()
        x = x.permute(0, 2, 1)  # torch.Size([706, 41, 256])
        # print('len(index_all) '+str(len(index_all))) #226
        # print('len(index_all[0]) '+str(len(index_all[0]))) #4
        # print('len(index_all[0])[0] '+str(len(index_all[0][0]))) #64

        # print(x.shape) #146x1x256
        x = x.cpu().detach().numpy()
        # print(x[1][0])
        i = 0
        mapping_x = []
        site_x = []
        while i < len(index_all):
            j = 0
            temp1 = []
            while j < len(index_all[i]):
                k = 0
                temp = []
                sitetemp = []
                while k < len(index_all[i][j]):
                    temp.append(x[i][index_all[i][j][k]])
                    k += 1
                sitetemp.append(x[i][20])
                temp1.append(temp)
                j += 1
            mapping_x.append(temp1)
            site_x.append(sitetemp)
            i += 1
        mapping_x = torch.tensor(mapping_x)  # torch.Size([706, 4, 10, 256])
        site_x = torch.tensor(site_x)  # torch.Size([706, 4, 10, 256])
        # print(mapping_x.shape)  # torch.Size([706, 4, 10, 256])
        # print(site_x.shape)  # torch.Size([706, 1, 256])
        return mapping_x, site_x

    def forward(self, index_all, x):
        x=x.to('cuda:0')
        x = x.permute(0, 2, 1)
        # print(x.shape) #torch.Size([706, 256, 41])
        x_site = self.cov1D_extracter1(x)
        # print(x.shape) #torch.Size([706, 256, 41])

        context_x, Asite_x = self.mapping_converge(x, index_all)
        context_x=context_x.to('cuda:0')
        Asite_x=Asite_x.to('cuda:0')

        x_bag = self.cov2D_extracter2(context_x).to('cuda:0')  # 120xkx(256/k)
        # print(x_bag.shape) #torch.Size([706, 4, 10, 256])

        return x_bag, Asite_x, x_site  # bx4x10x256, bx1x256, bx41x256


class SubGraphMapping(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(SubGraphMapping, self).__init__()
        #
        self.inputsize = inputsize  # 256
        self.outputsize = outputsize  # k
        self.featureMapping = FeatureConverge(self.inputsize, self.outputsize)
        # self.graph_embeddings=GraphConvSparse(260,260,activation=lambda x:x)
        """
        这部分的映射需要考虑在41个核苷酸之间进行筛选（为了保证筛选的合理合理性，选择不采样第21位的核苷酸，或者说采样（n+1）段）
        子图映射的工作：
        1.完成子图筛选(√)
        2.根据筛选出的子图，对特征进行汇聚成包（状态），其中，权重为多示例注意力权重，权重需要记录
        3.一维卷积
        4.逆向映射，通过共享权值对特征进行还原
        5.两种交叉熵Loss，包标签及其分类，结合位点标签及其分类。
        6（可考虑的）：除了生成包状态之外还生成对应的包嵌入。
        """

    def alias_setup(self, probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        # min_val = min(probs)
        # max_val = max(probs)
        #
        # probs = [(x - min_val) / (max_val - min_val) for x in probs]
        # print(probs)
        # print(K) #41
        q = np.zeros(K, dtype=np.float32)
        J = np.zeros(K, dtype=np.int32)

        smaller = []
        larger = []
        i = 0
        while i < len(probs):
            kk = i
            prob = probs[i]
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
            i += 1

        # print(smaller)
        # print(larger)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        # print('J '+str(J))
        # print('q '+str(q))

        return J, q

    def alias_draw(self, J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    # 采出4个子图，以将向量转换为145x4x64的形式
    def subgraphSelection(self, attention, degree):
        graph = attention.cpu().detach().numpy()
        graph_degree = degree.cpu().detach().numpy()

        # 选择四个权值最高的节点，计算两个方向上的核苷酸概率平均值，基于引入位置信息的alias-采样来选择子图节点
        # 在选择第一个子图之后，不对已经被选择的存在于子图中的节点进行挑选，因此需要重新规划概率列表
        # 返回采样出的位置列表.
        # 1.寻找最大度值点，返回位置
        import heapq
        s = 0
        all_all_index = []
        while s < len(graph):
            # print(s)
            # print(len(graph_degree[s]))
            # print(graph_degree[s])
            b = heapq.nlargest(self.outputsize, range(len(graph_degree[s])), graph_degree[s].take)  # 最大值所在的列表下标
            # print(b) #[2, 13, 14, 19]
            # print(b)

            # 2.根据最大值列表下标构造采样概率
            i = 0
            all_prob = []
            while i < len(b):
                hl = graph[s][b[i]][:]
                rl = graph[s][:][b[i]]
                # hrl = (hl + rl) * 100  # 扩充100倍，方便统计采样
                hrl = (hl + rl) / 10
                all_prob.append(hrl)
                i += 1

            # print(all_prob[0])
            # print(len(all_prob)) # 4
            # print(len(all_prob[0])) # 41
            # 3.根据列表选择采样的节点 (筛选概率选自度最大的节点对应的行，所有概率列表为[4x41])
            i = 0
            index_all = []
            selcted = []
            # print(int(self.inputsize/self.outputsize))
            while i < len(all_prob):  # 第一个维度为子图个数 不对修饰位点本身进行采样
                j = 0
                J, q = self.alias_setup(all_prob[i])  # 41 41
                # print(len(J))
                # print(len(q))
                temp = []
                counter = 0  # 计数器，防止无限遍历
                while j < int(41 / self.outputsize):  # 256,4
                    # print(j)
                    pre = j
                    sample = self.alias_draw(J, q)
                    if sample != 20:
                        # print(sample)
                        if counter < 10 and sample not in selcted:
                            temp.append(sample)
                            selcted.append(sample)
                            counter = 0
                            j += 1
                        if counter >= 10:
                            xx = 0
                            while xx < self.inputsize:
                                if xx not in selcted and xx != 20:
                                    temp.append(xx)
                                    selcted.append(xx)
                                    counter = 0
                                    j += 1
                                    break
                                xx += 1
                        if pre == j: counter += 1
                temp = np.sort(temp)
                index_all.append(temp)
                i += 1
            all_all_index.append(index_all)
            s += 1
        # print(all_all_index)
        # print(len(index_all[0]))
        # print(len(index_all[1]))
        # print(len(index_all[2]))
        # print(len(index_all[3]))
        # print(index_all[0])
        # print(index_all[1])
        # print(index_all[2])
        # print(index_all[3])
        # print('len(all_all_index[0]) '+str(len(all_all_index[0])))
        return all_all_index

    def forward(self, attention, degree, x):
        # print(attention.shape) #torch.Size([706, 41, 41])
        # print(degree.shape) #torch.Size([706, 41])
        index_all = self.subgraphSelection(attention, degree)  #

        # print('x1 '+str(x[0]))
        # x_embeddings=self.graph_embeddings(x)
        # print(x_embeddings.shape)

        x_bag, Asite_x, x_site = self.featureMapping(index_all, x)
        # print(x.shape) #120x10x26
        # print('x_bag.shape '+str(x_bag.shape)) #torch.Size([120, 10])
        # print('x_site.shape '+str(x_site.shape)) #torch.Size([120, 260])

        # print('x2 '+str(x[0]))
        return x_bag, Asite_x, x_site, index_all


class multiDomainSeqLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(multiDomainSeqLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.scale_size = 4  # 切比雪夫不等式的阶
        self.processing1 = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)  # （12x5,12x5)
        self.processing2 = nn.Linear(self.time_step * self.multi, self.time_step)  # (12x5,12)

        self.relu = nn.ReLU()
        self.Gating = nn.ModuleList()
        self.output_channel = self.scale_size * self.multi  # 20

    def forward(self, mul_L, x):
        mul_L = mul_L.unsqueeze(1)
        # print('mul_L.shape '+str(mul_L.shape))
        x = x.unsqueeze(1)
        # print('x.shape '+str(x.shape))
        spectralSeq = torch.matmul(mul_L, x).squeeze()
        spectralSeq = torch.sum(spectralSeq, dim=1)
        # print(spectralSeq.shape) #torch.Size([706, 41, 256])
        return spectralSeq


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2, device='cuda:0'):
        super(Model, self).__init__()
        self.unit = units  # 特征维度 100
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_graph = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))  # 1x12x1x1
        self.weight_key = nn.Parameter(torch.zeros(size=(41, 41)))  # 100x1 k
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(41, 41)))  # 100x1 q
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.ml = nn.Linear(self.time_step, self.unit)  # 输入维度（12）。输出维度（100）
        # self.GRU = nn.GRU(self.unit, self.unit) #输入维度（12）。输出维度（100）
        self.GRU = nn.GRU(41, 41)  # 输入维度（12）。输出维度（100）
        self.positioncode = TemporalConvNet(self.unit, self.unit)
        self.multi_layer = multi_layer
        self.k = 4

        self.seqGraphBlock = nn.ModuleList()  # ModuleList的优势：1、block的参数会自动加入到主模型中，且没有顺序性要求。
        self.seqGraphBlock.extend(
            [multiDomainSeqLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in
             range(self.stack_cnt)])

        self.subgraphmapping = SubGraphMapping(256, self.k)
        # self.fc = nn.Sequential( #input->target,12->3,100->100 用于约束模型输出和目标
        #     nn.Linear(int(self.time_step), int(self.time_step)),
        #     nn.LeakyReLU(),
        #     nn.Linear(int(self.time_step), self.horizon),
        # )

        # self.fc_site = nn.Sequential( #结合位点约束 12->12.100-100 增加sigmoid函数，同时返回节点位点和对应的权重 32x100x3
        #     nn.Linear(int(self.time_step), int(self.time_step)),
        #     nn.LeakyReLU(),
        #     nn.Linear(int(self.time_step), self.time_step),
        # )

        self.fc_shape = nn.Sequential(
            nn.Linear(int(self.unit), int(self.unit)),
            # nn.PReLU(),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(int(self.unit), int(self.unit)),
        )
        self.Ifc_shape = nn.Sequential(
            nn.Linear(int(self.unit), int(self.unit)),
            nn.Tanh(),
            nn.Linear(int(self.unit), 41),
        )
        self.fc_prob = nn.Sequential(
            nn.Linear(int(self.unit), self.unit),
            nn.Tanh(),
            nn.Linear(int(self.unit), 1),
        )

        self.fc_feature0 = nn.Sequential(
            nn.Linear(int(self.unit) , 64),
            # nn.PReLU(),
            nn.Tanh(),
        )

        self.fc_prob0 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


        self.fc_prob1 = nn.Sequential(
            nn.Linear(self.unit, 1),
            nn.Sigmoid(),
        )
        self.fc_feature1 = nn.Sequential(
            nn.Linear(int(self.unit) * 10, int(self.unit)),
            # nn.PReLU(),
            nn.Tanh(),
            nn.Linear(int(self.unit), 64),
            # nn.PReLU(),
            nn.Tanh(),
        )
        self.fc_prob1_1 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # self.fc_prob1_1 = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.Sigmoid(),
        # )

        # 考虑增加几层全连接层
        self.fc_feature2 = nn.Sequential(
            nn.Linear(int(self.unit) * 41, int(self.unit)),
            # nn.PReLU(),
            nn.Tanh(),
            nn.Linear(int(self.unit), 64),
            # nn.PReLU(),
            nn.Tanh(),
        )

        self.fc_prob2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(self.alpha)
        # self.relu = nn.PReLU()
        # self.relu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()
        # self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def cheb_polynomial(self, laplacian):  # 返回多阶拉普拉斯矩阵,这里使用的切比雪夫不等式的四阶式子
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        # print('laplacian.shape '+str(laplacian.shape)) #100x100
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    # 为每一条序列都计算一个拉普拉斯矩阵
    def cheb_polynomial_multi(self, laplacian):  # 返回多阶拉普拉斯矩阵,这里使用的切比雪夫不等式的四阶式子
        # print('laplacian.shape '+str(laplacian.shape)) #torch.Size([145, 41, 41])
        bat, N, N = laplacian.size()  # [N, N] 512
        laplacian = laplacian.unsqueeze(1)
        first_laplacian = torch.zeros([bat, 1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=1)

        return multi_order_laplacian  # 32x12x4x100x100

    def seq_graph_ing(self, x):

        input = self.positioncode(x.contiguous())
        # input =input.repeat(1,1,256) #考虑更复杂的数据处理方式
        input = torch.matmul(input.permute(0, 2, 1), input)  # 考虑更复杂的数据处理方式
        # 考虑增加一个标准化
        # normalized_tensor = F.normalize(input, p=2, dim=2)
        # print(normalized_tensor[0])

        # 暂时使用这种最大最小归一化
        min_vals = input.min(dim=1, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_vals = input.max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        normalized_tensor = (input - min_vals) / (max_vals - min_vals)
        # print(normalized_tensor[0][1])

        input, _ = self.GRU(input)  # 加快收敛

        attention = self.district_graph_attention(input)  # 32x100x100 attention通过自注意力机制得到的，此时的attention被当成了一个图
        attention_all = attention

        degree_all = torch.sum(attention, dim=2)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)  # 求度矩阵，按列相加 torch.size([140])
        attention = 0.5 * (attention + attention.T)  # 转成对称矩阵
        degree_l = torch.diag(degree)  # 度的对角矩阵
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 0.1))

        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))  # 得到拉普拉斯矩阵，类似GCN

        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention_all, degree_all  # 返回一个多阶的拉普拉斯矩阵，以及一个注意力矩阵（100x100）

    def seq_graph_ing_multi(self, x):
        x=x.to('cuda:0')
        input = self.positioncode(x.contiguous()).to('cuda:0')
        # input =input.repeat(1,1,256) #考虑更复杂的数据处理方式
        input = torch.matmul(input.permute(0, 2, 1), input)  # 考虑更复杂的数据处理方式
        # 考虑增加一个标准化
        # normalized_tensor = F.normalize(input, p=2, dim=2)
        # print(normalized_tensor[0])

        # 暂时使用这种最大最小归一化
        # min_vals = input.min(dim=1, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        # max_vals = input.max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        # normalized_tensor = (input - min_vals) / (max_vals - min_vals)
        # print(normalized_tensor[0][1])

        # input, _ = self.GRU(input)  # 加快收敛
        input, _ = self.GRU(input)  # 加快收敛

        attention = self.district_graph_attention(input) # 32x100x100 attention通过自注意力机制得到的，此时的attention被当成了一个图
        attention_all = attention

        degree_all = torch.sum(attention, dim=2)
        # attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=-1)  # 求度矩阵，按列相加 torch.size([140])
        # print(degree.shape) #torch.Size([706, 41])
        attention = 0.5 * (attention + attention.permute(0, 2, 1))  # 转成对称矩阵
        degree_l = tensor_diag(degree).to('cuda:0')  # 度的对角矩阵
        diagonal_degree_hat = tensor_diag(1 / (torch.sqrt(degree) + 1e-6)).to('cuda:0')

        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))  # 得到所有序列对应的拉普拉斯矩阵

        mul_L = self.cheb_polynomial_multi(laplacian)
        return mul_L, attention_all, degree_all  # 返回一个多阶的拉普拉斯矩阵，以及一个注意力矩阵（100x100）

    def district_graph_attention(self, input):

        # Q和 K 两个矩阵本身包含共识性，可以考虑可视化这两个矩阵，也可以考虑分别可视化attention矩阵
        key = torch.matmul(input, self.weight_key)  # 32x100x1
        query = torch.matmul(input, self.weight_query)  # 32x100x1
        data = query * key.permute(0, 2, 1)
        attention = self.relu(data)
        # attention = F.softmax(data, dim=1) # 暂时不考虑这个
        # 这部分可以考虑，对原始的【-1,41,256】的矩阵进行按行平均，然后与自注意力矩阵做哈达玛乘积
        # print(attention.shape) #torch.Size([706, 41, 41])
        return attention

    def reweight(self, x, y, z):
        weight_all = []
        # print(y.squeeze().shape) #x:145x4x10 y:145x4
        # print(y.shape) # torch.Size([706, 4, 10])
        # print(z.shape) # torch.Size([706])
        y = y.squeeze().cpu().detach().numpy()
        z = z.squeeze().cpu().detach().numpy()
        i = 0
        while i < len(x):
            j = 0
            temp_weight = [0] * 41
            while j < len(x[i]):
                k = 0
                while k < len(x[i][j]):
                    temp_weight[x[i][j][k]] = y[i][j][k]
                    k += 1
                j += 1
            temp_weight[20] = z[i]
            weight_all.append(temp_weight)
            i += 1
        return torch.tensor(weight_all)

    def forward(self, x):
        # print(x.shape) #torch.Size([145, 1, 41])
        x=x.to('cuda:0')
        x = self.fc_shape(x)  # torch.Size([-1, 41, 256])
        mul_L, attention, degree = self.seq_graph_ing_multi(x)  # attention(-1,256,256) degree(-1,256)
        # print(mul_L.shape) # torch.Size([4, 41, 41]),当保留平均时，该矩阵可以认为是所有核苷酸关联的共识矩阵
        # print(mul_L.shape) # torch.Size([706, 4, 41, 41]),当不保留平均时，该矩阵可以认为是所有核苷酸关联的共识矩阵
        # X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        X = x.unsqueeze(1).contiguous()  # torch.Size([706, 1, 41, 256])
        # print(X.shape)  # torch.Size([706, 1, 256, 41])
        result = []
        for stack_i in range(self.stack_cnt):  # stack_i判断进入模块中的哪一块
            forecast = self.seqGraphBlock[stack_i](mul_L, X)
            result.append(forecast)
            # print(stack_i)
        forecast = result[0] + result[1]  # torch.Size([706, 41, 256]) 做了一个残差连接

        x_bag, Asite_x, x_site, index_all = self.subgraphmapping(attention, degree,
                                                                 forecast)  # 返回值分别为局部表征，修饰位点表征，全局表征和核苷酸索引

        # forecast_site_prob = forecast
        #
        # forecast_feature = forecast_site_prob.permute(0, 2, 1).contiguous().view(-1, self.unit) # 卷积后的表征

        '''
        关于局部嵌入（x_bag）：1. 计算局部权重。2. 计算局部嵌入的识别分数。 
        '''
        forecast_site_prob_bag = self.fc_prob1(x_bag)  # 1. 局部权重 706x4x10x1
        x_bag_new = x_bag.reshape(-1, self.k, x_bag.shape[-2] * x_bag.shape[-1])
        x_bag_new = self.fc_feature1(x_bag_new)
        forecast_site_score_bag = self.fc_prob1_1(x_bag_new)  # 2. 识别分数 706x4x1

        '''
        关于修饰位点的自身嵌入（Asite_x）：1. 计算局部权重。2. 计算局部嵌入的识别分数。 两者相同。
        '''
        Asite_x=self.fc_feature0(Asite_x)
        forecast_Asite_prob_score = self.fc_prob0(Asite_x)  # 1. 局部权重 706x1x1

        '''
        关于全局嵌入（x_site，未整合）：1. 计算全局嵌入的识别分数。 
        '''
        # 全局嵌入
        x_site=x_site.permute(0, 2, 1)
        x_site_global = x_site.reshape(-1, x_site.shape[-1] * x_site.shape[-2])
        x_site_global = self.fc_feature2(x_site_global)  # 706x64
        # print(x_site_global.shape)
        forecast_site_prob = self.fc_prob2(x_site_global)  # 1. 全局嵌入的识别分数 706x1,未整合

        '''
        关于整合后的嵌入（x_site，未整合）：1. 计算全局嵌入的识别分数。 
        '''
        forecast_site_prob_bag = forecast_site_prob_bag.squeeze()
        forecast_Asite_prob_score = forecast_Asite_prob_score.squeeze()
        weight_Local = self.reweight(index_all, forecast_site_prob_bag, forecast_Asite_prob_score).to('cuda:0')
        # print(weight_Local.shape) #torch.Size([706, 41])
        x_feature_combine = x_site * torch.unsqueeze(weight_Local, -1)  # 整合后特征
        x_feature_combine = self.fc_feature2(x_feature_combine.reshape(-1,x_feature_combine.shape[-1]*x_feature_combine.shape[-2]))
        x_feature_combine_score = self.fc_prob2(x_feature_combine)

        '''
        返回值分别为：
        1. 四个识别分数：（1）未整合的全局嵌入的识别分数（706x1）。（2）整合后的全局嵌入的识别分数（706x1）。（3）局部嵌入的识别分数（706x4x1）。（4）单个修饰位点的识别分数（706x1）
        2. 四种序列表征：（1）未整合的全局嵌入（706x64）。（2）整合后的全局嵌入（706x64）。（3）局部嵌入（706x4x64）。（4）单个修饰位点的嵌入（706x64）
        3. 每个序列上的核苷酸权重，用于可解释(706x41)
        '''
        forecast_site_prob=forecast_site_prob.squeeze()
        x_feature_combine_score=x_feature_combine_score.squeeze()
        forecast_site_score_bag=forecast_site_score_bag.squeeze()
        forecast_Asite_score = forecast_Asite_prob_score.squeeze()

        Asite_x=Asite_x.squeeze()

        return forecast_site_prob, x_feature_combine_score, forecast_site_score_bag, forecast_Asite_score, \
               x_site_global, x_feature_combine, x_bag_new, Asite_x, \
               weight_Local
