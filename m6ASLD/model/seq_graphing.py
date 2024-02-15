import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
from .tcn import *
# from .Causal_Dilate_Network import *
import pandas as pd
import csv


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
        out = self.net(x)
        # print(out.shape)
        # print('out.shape '+str(out.shape)) #out.shape torch.Size([16, 600, 80])
        # res = x if self.downsample is None else self.downsample(x)
        # print('self.relu(out + res).shape '+ str(self.relu(out + res).shape)) #torch.Size([16, 600, 80])
        return self.tanh1(out)


class IMILmask(nn.Module):
    def __init__(self):
        super(IMILmask, self).__init__()

        self.L1 = nn.Sequential(
            nn.Linear(64, 64),
            # nn.relu()
            # nn.relu()
            nn.ReLU()
            # nn.Tanh()
            # nn.LeakyReLU()
        )

        self.L2 = nn.Sequential(
            nn.Linear(64, 64),
            # nn.Tanh()
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(64, 64)  # 注意力权重
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
    def __init__(self):
        super(FeatureConverge, self).__init__()
        # self.inputsize=inputsize
        # self.outputsize=outputsize
        self.cov1D_extracter1 = Conv1D_feature_extracter(1, 1, 3, 1, 1, 0.2)
        self.mask = IMILmask()
        self.cov1D_extracter2 = Conv1D_feature_extracter(4, 4, 3, 1, 1, 0.2)  # 通过1维卷积将其转为包状态
        # 对输入序列进行执行一维卷积
        # 计算注意力权重
        # 生成两个分支，一个分支为包嵌入的包分类（120x10x1），一个分支为包内结合位点的分类(120x260x1)
        return

    def mapping_converge(self, x, index_all):  # 根据采样获得的下标
        # index_all 4x64的列表
        a, b, c = x.size()
        # print(x.shape) #146x1x256
        x = x.detach().numpy()
        # print(x[1][0])
        i = 0
        mapping_x = []
        while i < a:
            j = 0
            temp1 = []
            while j < len(index_all):
                k = 0
                temp = []
                while k < len(index_all[j]):
                    temp.append(x[i][0][index_all[j][k]])
                    k += 1
                temp1.append(temp)
                j += 1
            mapping_x.append(temp1)
            i += 1
        mapping_x = torch.tensor(mapping_x)
        # print(mapping_x.shape) #146x4x64
        return mapping_x

    def forward(self, index_all, x):
        # print('x_pre.shape '+str(x.shape))
        # 需要重新考虑包嵌入的角色，若
        x = x.permute(0, 2, 1)
        x_site = self.cov1D_extracter1(x)
        # print('x_IMIL_pre.shape '+str(x.shape)) #154x4x64
        x = self.mapping_converge(x, index_all)
        # x=torch.reshape(x_site,(-1,4,64)) #reshape
        weight_A, x = self.mask(x)
        # print('weight_A.shape '+str(weight_A.shape)) #120x4x64
        x_bag = self.cov1D_extracter2(x)  # 120x4x64
        # x_bag=self.fc_1(x)
        # print('x_bag.shape '+str(x_bag.shape)) #120x10x1
        # x_site=torch.mul(weight_A,x_bag)
        # x_site=torch.mul(weight_A,x_bag)
        # x_site=torch.reshape(x_site,(-1,260))
        # x_site=torch.reshape(x_site,(-1,256))
        # print(x_bag.shape)
        # print(weight_A)
        # 根据包的simoid结果，逆向对包内的结合位点进行投影

        # x_site=torch.sigmoid(x_site)
        return x_bag, x_site  # 145x4x64,145x1x256


class SubGraphMapping(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(SubGraphMapping, self).__init__()
        #
        self.inputsize = inputsize  # 256x256
        self.outputsize = outputsize  # 256x1
        self.featureMapping = FeatureConverge()
        # self.graph_embeddings=GraphConvSparse(260,260,activation=lambda x:x)
        """
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

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

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
        graph = attention.detach().numpy()
        graph_degree = degree.detach().numpy()

        # 选择四个权值最高的节点，计算两个方向上的核苷酸概率平均值，基于引入位置信息的alias-采样来选择子图节点
        # 在选择第一个子图之后，不对已经被选择的存在于子图中的节点进行挑选，因此需要重新规划概率列表
        # 返回采样出的位置列表.
        # 1.寻找最大度值点，返回位置
        import heapq
        b = heapq.nlargest(4, range(len(graph_degree)), graph_degree.take)  # 最大值所在的列表下标
        # print(b)

        # 2.根据最大值列表下标构造采样概率
        i = 0
        all_prob = []
        while i < len(b):
            hl = graph[b[i]][:]
            rl = graph[:][b[i]]
            hrl = (hl + rl) * 100  # 扩充100倍，方便统计采样
            all_prob.append(hrl)
            i += 1
        # print(all_prob[0])
        # 3.根据列表选择采样的节点
        i = 0
        index_all = []
        selcted = []
        while i < len(all_prob):
            j = 0
            J, q = self.alias_setup(all_prob[i])
            temp = []
            while j < 64:
                sample = self.alias_draw(J, q)
                if sample not in selcted:
                    temp.append(sample)
                    selcted.append(sample)
                    j += 1
            temp = np.sort(temp)
            index_all.append(temp)
            i += 1
        #
        # print(len(index_all[0]))
        # print(index_all[1])
        # print(index_all[2])
        # print(index_all[3])
        return index_all

    def forward(self, attention, degree, x):
        index_all = self.subgraphSelection(attention, degree)
        StorFile(index_all, 'index_all.csv')
        # print('index_all '+str(index_all))
        # print('x1 '+str(x[0]))
        # x_embeddings=self.graph_embeddings(x)
        # print(x_embeddings.shape)

        x_bag, x_site = self.featureMapping(index_all, x)
        # print(x.shape) #120x10x26
        # print('x_bag.shape '+str(x_bag.shape)) #torch.Size([120, 10])
        # print('x_site.shape '+str(x_site.shape)) #torch.Size([120, 260])

        # print('x2 '+str(x[0]))
        return x_bag, x_site


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

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        # print('mul_L.shape '+str(mul_L.shape)) #4, 1, 256, 256
        x = x.unsqueeze(1)

        # print('x.shape '+str(x.shape)) #145, 1, 1, 256, 1
        # print('mul_L.shape '+str(mul_L.shape))
        spectralSeq = torch.matmul(mul_L, x)
        # print("spectralSeq.shape "+str(spectralSeq.shape))
        # HightSpectralSeq = self.HighFrequencyRetentionBlock(spectralSeq).unsqueeze(2)
        HightSpectralSeq = spectralSeq.repeat(1, 1, 1, 1, 5)  # 考虑更复杂的数据处理方式
        HightSpectralSeq = torch.sum(HightSpectralSeq, dim=1)  # mean or sum
        # print(HightSpectralSeq.shape) #torch.Size([145, 1, 256, 5])

        # 观察修改这两种处理方式后的实验结果
        BindingSiteSeq0 = self.processing1(HightSpectralSeq).squeeze(1)
        BindingSiteSeq = self.processing2(BindingSiteSeq0)
        # print('BindingSiteSeq.shape '+str(BindingSiteSeq.shape)) #BindingSiteSeq.shape

        return BindingSiteSeq


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 # units:输出维度100 stack_cnt：块数2 time_step:window size 12
                 # multi_layer 层数 5
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units  # 特征维度 100
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_graph = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))  # 1x12x1x1
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))  # 100x1 k
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))  # 100x1 q
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.ml = nn.Linear(self.time_step, self.unit)  # 输入维度（12）。输出维度（100）
        self.GRU = nn.GRU(self.unit, self.unit)  # 输入维度（12）。输出维度（100）
        self.positioncode = TemporalConvNet(self.unit, self.unit)
        self.multi_layer = multi_layer

        self.seqGraphBlock = nn.ModuleList()  # ModuleList的优势：1、block的参数会自动加入到主模型中，且没有顺序性要求。
        self.seqGraphBlock.extend(
            [multiDomainSeqLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in
             range(self.stack_cnt)])

        self.subgraphmapping = SubGraphMapping(256, 4)
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
            nn.Linear(41, int(self.unit)),
            # nn.LeakyReLU(),
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
        self.fc_prob1 = nn.Sequential(
            nn.Linear(int(self.unit / 4), 1),
            nn.Sigmoid(),
        )
        # self.fc_prob1_1 = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.Sigmoid(),
        # )
        self.fc_prob2 = nn.Sequential(
            nn.Linear(int(self.unit), 1),
            nn.Sigmoid(),
        )
        # self.relu = nn.ReLU(self.alpha)
        self.relu = nn.LeakyReLU(self.alpha)
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

    def seq_graph_ing(self, x, input_prob):

        input = self.positioncode(x.contiguous())
        input = input.repeat(1, 1, 256)  # 考虑更复杂的数据处理方式
        input, _ = self.GRU(input)  # 加快收敛，生成特征图。

        attention = self.district_graph_attention(input,
                                                  input_prob)  # 32x100x100 attention通过自注意力机制得到的，此时的attention被当成了一个图

        attention = torch.mean(attention, dim=0)  # 100x100 此处得到的100x100矩阵即为描述特征i和j关系的矩阵，每个时间戳之间的相关性？
        degree = torch.sum(attention, dim=1)  # 求度矩阵，按列相加 torch.size([140])
        attention = 0.5 * (attention + attention.T)  # 转成对称矩阵
        degree_l = torch.diag(degree)  # 度的对角矩阵
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 0.1))

        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))  # 得到拉普拉斯矩阵，类似GCN

        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention, degree  # 返回一个多阶的拉普拉斯矩阵，以及一个注意力矩阵（100x100）

    def get_graph_sequence(self, attention, input_prob):

        # print(attention.shape) #torch.Size([145, 256, 256])
        # print(input_prob.shape) #torch.Size([145, 1, 256])

        attention_groups = torch.chunk(attention, 29, dim=0)
        probs_groups = torch.chunk(input_prob.squeeze(), 29, dim=0)

        graphs = attention_groups[0]
        probs = probs_groups[0]

        graphs = list(graphs.detach().numpy())

        xx = 0
        while xx < len(graphs):
            graphs[xx] = list(graphs[xx])
            yy = 0
            while yy < len(graphs[xx]):
                graphs[xx][yy] = list(graphs[xx][yy])
                yy += 1
            xx += 1

        yy = 0
        # print(probs.shape)
        probs = list(probs.detach().numpy())  # 5x256

        while yy < len(probs):
            probs[yy] = list(probs[yy])
            yy += 1

        return graphs[0], graphs[1], graphs[2], graphs[3], graphs[4], probs

    def district_graph_attention(self, input, input_prob):
        input = input.permute(0, 2, 1).contiguous()  # 32x100x100
        bat, N, fea = input.size()  # 32 140 140
        key = torch.matmul(input, self.weight_key)  # 32x100x1
        query = torch.matmul(input, self.weight_query)  # 32x100x1
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.relu(data)
        attention = F.softmax(data, dim=2)  # 该100x100为时间戳时间戳之间的矩阵，即
        input_prob = normalized_input(input_prob)
        L = torch.mean(input_prob, 1)
        L = L.unsqueeze(1)
        # attention=attention
        attention = attention * (L)
        # attention = self.dropout(attention)
        xx = attention.squeeze().detach().numpy()
        # 存储前2个图和前2个input prob
        graph0, graph1, graph2, graph3, graph4, probs = self.get_graph_sequence(attention, input_prob)

        StorFile(graph0, 'all_graphs0.csv')
        StorFile(graph1, 'all_graphs1.csv')
        StorFile(graph2, 'all_graphs2.csv')
        StorFile(graph3, 'all_graphs3.csv')
        StorFile(graph4, 'all_graphs4.csv')
        # StorFile(probs,'all_probs.csv')
        # StorFile(probs,'all_probs.csv')
        # print("attention.shape "+str(attention.shape))
        # StorFile(xx,'attention.csv')

        return attention

    def forward(self, x, input_prob):

        print('x.shape ' + str(x.shape))  # 145x1x41
        x = self.fc_shape(x)
        print('x.shape ' + str(x.shape))  # 145x1x256
        input_prob = self.fc_shape(input_prob)  # 145x1x256
        mul_L, attention, degree = self.seq_graph_ing(x, input_prob)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):  # stack_i判断进入模块中的哪一块
            forecast = self.seqGraphBlock[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]  # torch.size([32,100,12]) 做了一个残差连接
        # forecast = result[0] #torch.size([32,100,12]) 做了一个残差连接
        # forecast = result[1] #torch.size([32,100,12]) 做了一个残差连接
        # forecast_all=forecast
        # forecast = self.fc(forecast) #torch.size([32,20,3])
        # forecast_site_prob=torch.sigmoid(forecast_all) #不增加全连接层和LeakyreLu函数

        # print('attention.shape '+str(attention.shape)) #256x256
        x_bag, x_site = self.subgraphmapping(attention, degree, forecast)

        # print('x_bag.shape '+str(x_bag.shape)) #x_bag.shape torch.Size([145, 4, 64])
        # print('x_site.shape '+str(x_site.shape)) #x_site.shape torch.Size([145, 256])

        forecast_site_prob = forecast
        # print(forecast_site_prob[0])
        forecast_feature = forecast_site_prob.permute(0, 2, 1).contiguous().view(-1, self.unit)
        # print('forecast_feature.shape '+str(forecast_feature.shape))
        # StorFile(forecast_feature.detach().numpy(), '1307_feature/Train_Test_final_feature' + str(epoch) + '.csv')

        # print('x_site.shape ' + str(x_site.shape))
        # forecast_site_prob=forecast_site_prob.permute(0, 2, 1)
        # print(forecast_site_prob.permute(0, 2, 1).shape)
        # forecast_site_prob=torch.sigmoid(self.fc_prob(forecast_site_prob.permute(0, 2, 1))).contiguous()
        forecast_site_prob_bag = self.fc_prob1(x_bag)
        # print(forecast_site_prob_bag.shape) #145x4x1
        forecast_site_prob = self.fc_prob2(x_site)
        # forecast_site_prob=torch.sigmoid(torch.sum(forecast_site_prob,dim=)).contiguous()
        # print('forecast_site_prob.shape '+str(forecast_site_prob.shape))

        # 增加一个拼接操作，特征张量x_bag 145x4x64,特征分区权重为forecast_site_prob_bag 145x4x1 直接向量乘法？
        x_feature_combine = torch.mul(torch.reshape(x_site, (-1, 4, 64)), (forecast_site_prob_bag))
        # x_feature_combine=torch.mul(x_bag,(forecast_site_prob_bag))

        # x_feature_combine=torch.mul(torch.reshape(x_site,(-1,4,64)),1)
        x_feature_combine = torch.reshape(x_feature_combine, (-1, 256))

        x_feature_combine_result = self.fc_prob2(x_feature_combine)

        forecast_site_prob = torch.squeeze(forecast_site_prob, 2)
        forecast_site_prob_bag = torch.squeeze(forecast_site_prob_bag, 2)  # 特征上的分区权重
        # x_feature_combine_result=torch.squeeze(x_feature_combine_result,2) #特征上的分区权重
        # print(x_feature_combine_result.shape)

        # forecast_site_prob_bag=self.fc_prob1_1(forecast_site_prob_bag) #计算拼接权重
        # print('forecast.shape '+str(forecast.shape))

        return forecast_site_prob, forecast_site_prob_bag, x_feature_combine_result, forecast_feature
