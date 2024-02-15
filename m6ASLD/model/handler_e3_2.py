import json
from datetime import datetime
import warnings

from data_loader.SiteBinding_dataloader0 import ForecastDataset
from .seq_graphing_e3_2 import Model
# from models.seq_graph import Model

import torch.utils.data as torch_data
import time
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from .Utils import *

from utils.math_utils import evaluate
import random


fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

warnings.filterwarnings("ignore")

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def save_model(model, model_dir, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # epoch = str(epoch) if epoch else ''
    fold = str(fold) if fold else ''
    file_name = os.path.join(model_dir, fold + '_PepBindA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch)
    file_name = os.path.join(model_dir, epoch + '_PepBindA.pt')
    print(file_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (
        inputs, inputs_site, input_prob, target, target_site, target_prob, train_info, target_info) in enumerate(
                dataloader):
            inputs = inputs  # 输入cgr序列
            target = target  # 目标CGR序列

            #对输入的序列数据进行标准化
            # inputs= normalized_input(inputs)
            # target= normalized_input(target)
            # inputs=normalize_input(inputs)
            # target=normalize_input(target)
            # inputs,_= normalized(inputs, 'z_score')
            # target,_= normalized(target, 'z_score')

            # print('inputs_inference.shape '+str(inputs.shape))
            # print('target_inference.shape '+str(target.shape))

            inputs_site = inputs_site  # 输入结合位点
            target_site = target_site  # 目标结合位点

            input_prob = input_prob  # 输入的拉普拉斯值
            target_prob = target_prob  # 目标的拉普拉斯值

            train_info = train_info  # 输入的相关信息，包括窗口所在序列、窗口中实际、亲和力的值
            target_info = target_info

            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result, a ,forecast= model(inputs,input_prob)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, dataloader, device,
             node_cnt, window_size, horizon,
             result_file=None):
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    # if normalize_method and statistic:
    #     forecast = de_normalized(forecast_norm, normalize_method, statistic)
    #     target = de_normalized(target_norm, normalize_method, statistic)
    # else:
    forecast, target = forecast_norm, target_norm
    score = evaluate(target, forecast)
    score_by_node = evaluate(target, forecast, by_node=True)
    end = datetime.now()

    score_norm = evaluate(target_norm, forecast_norm)
    # print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2])

def validate_inference_binding_site(model, dataloader,k):
    model.eval()
    with torch.no_grad():
        for i, (inputs, inputs_labels) in enumerate(dataloader):
            inputs = inputs  # 输入cgr序列
            inputs_labels = inputs_labels  # 输入结合位点
            # inputs = normalized_input(inputs)
            # print(inputs.shape)  # torch.Size([32, 12, 51])
            # print(inputs_labels.shape)  # 32x12
            '''
            返回值分别为：
            1. 四个识别分数：（1）未整合的全局嵌入的识别分数（706x1）。（2）整合后的全局嵌入的识别分数（706x1）。（3）局部嵌入的识别分数（706x4x1）。（4）单个修饰位点的识别分数（706x1）
            2. 四种序列表征：（1）未整合的全局嵌入（706x64）。（2）整合后的全局嵌入（706x64）。（3）局部嵌入（706x4x64）。（4）单个修饰位点的嵌入（706x64）
            3. 每个序列上的核苷酸权重，用于可解释(706x41)
            '''
            forecast_site_score, x_feature_combine_score, forecast_site_score_bag, forecast_Asite_score, \
            x_site_global, x_feature_combine, x_bag_new, Asite_x, \
            weight_Local= model(inputs)

            # print(forecast_result)
            # prediction,label,_,_=Sequence_reduction(inputs_site,forecast_result,train_info)
            # prediction,label,_,_=Sequence_reduction(inputs_site,forecast_result,train_info)
            # print('forecast_result.shape '+str(forecast_result.shape))
            # result,Real_Prediction,Real_Prediction_Prob=Indicator(inputs_labels,forecast_result)
            result,Real_Prediction,Real_Prediction_Prob=Indicator(inputs_labels,x_feature_combine_score)

            #
            validate_auc, _, _ = auroc(x_feature_combine_score, inputs_labels)
            validate_aupr, _, _ = auprc(x_feature_combine_score, inputs_labels)
            # print('validate_auc: '+str(validate_auc)+' '+'validate_aupr: '+str(validate_aupr))
            result[2]=round(validate_aupr,4)

            labels_real = list(inputs_labels.contiguous().view(-1).cpu().detach().numpy())

            # 全局特征
            forecast_feature = list(x_feature_combine.cpu().detach().numpy()) #全局特征
            xx = 0
            while xx < len(forecast_feature):
                forecast_feature[xx]=list(forecast_feature[xx])
                forecast_feature[xx].append(int(labels_real[xx]))
                xx += 1

            #两部分内容：1）存储四类依赖模式的特征。2）存储四类依赖模式的结果
            x_bag_f_4=torch.chunk(x_bag_new,k,dim=1)
            xxx=0
            validate_bag_feature=[]
            while xxx<k:
                validate_bag_feature.append(list(x_bag_f_4[xxx].squeeze().cpu().detach().numpy()))
                xxx+=1
            x_bag_tag_4=torch.chunk(forecast_site_score_bag,k,dim=1)

            All_bag_result=[]
            All_bag_rp=[]
            All_bag_rpp=[]

            xxx=0
            while xxx<k:
                result0, rp0, rpp0 = Indicator1(inputs_labels, x_bag_tag_4[xxx])
                All_bag_result.append(result0)
                All_bag_rp.append(rp0)
                All_bag_rpp.append(rpp0)
                xxx+=1
            # print('inputs_labels '+str(inputs_labels.shape))
            # print('inputs_labels '+str(inputs_labels.shape))

            validate_bag_feature_new=[]
            while xxx < k:
                feature0 = validate_bag_feature[xxx]
                xx = 0
                while xx < len(feature0):
                    feature0[xx] = list(feature0[xx])
                    feature0[xx].append(int(labels_real[xx]))
                    xx += 1
                validate_bag_feature_new.append(feature0)
                xxx += 1

            # print(feature3[0])
        # mean_validate_aupr=aupr_total/cnt
    # print('validate_auc0: '+str(mean_validate_auc)+' '+' validate_aupr0: '+str(mean_validate_aupr))
    return  result,Real_Prediction,Real_Prediction_Prob,forecast_feature,\
            validate_bag_feature_new,All_bag_result,All_bag_rp,All_bag_rpp\

def train(train_data, valid_data, args,result_file, species, fold):
    # node_cnt = int((train_data.shape[1]-3)/3) #100 (固定窗口大小或自适应窗口大小)
    node_cnt = 256
    K_sub=4
    print('node_cnt '+str(node_cnt))
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon,device=args.device)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data)
    valid_set = ForecastDataset(valid_data)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    # 对训练集里的每个类别加一个权重。如果该类别的样本数多，那么它的权重就低，反之则权重就高
    criterion = torch.nn.BCELoss( reduction='mean')  # 计算目标值和预测值之间的二进制交叉熵损失函数
    focal_loss=FocalLoss()

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_Acc= 0.0
    best_result=[]
    best_Real_Predition=[]
    best_Real_Predition_Prob=[]
    best_train_feature=[]
    best_validate_feature=[]
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        auc_total=0
        aupr_total=0

        auc_total_combine = 0
        aupr_total_combine = 0


        auc_total_bag=[0]*K_sub
        aupr_total_bag=[0]*K_sub
        Temp_train_feature=[]
        Temp_train_weight_score=[]
        Temp_train_bag_feature=[]


        for i, (
        inputs, inputs_labels) in enumerate(
                train_loader):
            inputs = inputs.to(args.device)  # 输入cgr序列


            inputs_labels = inputs_labels.to(args.device)  # 输入结合位点

            '''
            返回值分别为：
            1. 四个识别分数：（1）未整合的全局嵌入的识别分数（706x1）。（2）整合后的全局嵌入的识别分数（706x1）。（3）局部嵌入的识别分数（706x4x1）。（4）单个修饰位点的识别分数（706x1）
            2. 四种序列表征：（1）未整合的全局嵌入（706x64）。（2）整合后的全局嵌入（706x64）。（3）局部嵌入（706x4x64）。（4）单个修饰位点的嵌入（706x64）
            3. 每个序列上的核苷酸权重，用于可解释(706x41)
            '''
            forecast_site_score, x_feature_combine_score, forecast_site_score_bag, forecast_Asite_score, \
            x_site_global, x_feature_combine, x_bag_new, Asite_x, \
            weight_Local= model(inputs) #32x12x100 结合位点

            # forecast_site_prob_bags=torch.split(forecast_site_prob_bag,4,-1)

            labels_real = list(inputs_labels.contiguous().view(-1).cpu().detach().numpy())

            # 局部评分存储
            weight_Local = list(weight_Local.cpu().detach().numpy())
            xx = 0
            while xx < len(weight_Local):
                weight_Local[xx] = list(weight_Local[xx])
                weight_Local[xx].append(int(labels_real[xx]))
                xx += 1
            Temp_train_weight_score.extend(weight_Local)

            # 全局特征存储
            x_feature_combine = list(x_feature_combine.cpu().detach().numpy())
            xx = 0
            while xx < len(x_feature_combine):
                x_feature_combine[xx]=list(x_feature_combine[xx])
                x_feature_combine[xx].append(int(labels_real[xx]))
                xx += 1
            Temp_train_feature.extend(x_feature_combine)

            # 局部特征存储
            forecast_bag_feature = list(x_bag_new.cpu().detach().numpy())
            xx = 0
            forecast_bag_feature_new=[]
            while xx < len(forecast_bag_feature):
                yy=0
                forecast_bag_feature[xx]=list(forecast_bag_feature[xx])
                while yy<len(forecast_bag_feature[xx]):
                    forecast_bag_feature[xx][yy]=list(forecast_bag_feature[xx][yy])
                    forecast_bag_feature[xx][yy].append(int(labels_real[xx]))
                    # forecast_bag_feature_new.append(list(temp))
                    yy+=1
                xx += 1
                # print(len(forecast_bag_feature_new))
            Temp_train_bag_feature.extend(forecast_bag_feature)


            # StorFile(forecast_feature, 'mm_Pse_472_feature/Train_Test_final_feature'+'_' + str(epoch)+'_'+str(i) + '.csv')
            # StorFile(forecast_feature,
            #          'mm_m6A_725/Train_Test_final_feature' + '_' + str(epoch) + '_' + str(i) + '.csv')
            # pd.DataFrame(forecast_feature.cpu().detach().numpy()).to_csv('Train_Test_final_featurex' + str(epoch) + '.csv',header=None,index=None)
            #此处增加mask，即重新调整
            # scale=0.5
            # _,prob,site=Index_Mask(forecast_site_prob,inputs_site)

            # forecast_site_label = Prediction_label(forecast_site_prob).contiguous()
            train_auc,_,_=auroc(forecast_site_score,inputs_labels)
            train_aupr,_,_=auprc(forecast_site_score,inputs_labels)

            train_auc_combine,_,_=auroc(x_feature_combine_score,inputs_labels)
            train_aupr_combine,_,_=auprc(x_feature_combine_score,inputs_labels)

            bags = torch.chunk(forecast_site_score_bag, K_sub, dim=1)
            result_auc_bag=[]
            for ele in bags:
                # print(ele.shape)
                train_auc_bag,_,_=auroc(ele,inputs_labels)
                result_auc_bag.append(train_auc_bag)

            # print(bags[0].shape)
            result_aupr_bag = []
            for ele in bags:
                train_aupr_bag,_,_=auprc(ele,inputs_labels)
                result_aupr_bag.append(train_aupr_bag)

            binding_loss = criterion(forecast_site_score, inputs_labels.float())
            combine_binding_loss = criterion(x_feature_combine_score, inputs_labels.float())


            bag_loss=0
            j=0
            while j<len(bags):
                bag_loss+=criterion(torch.tensor(bags[j]).squeeze(), inputs_labels.float())
                j+=1
            bag_loss=bag_loss/len(bags)

            all_loss=combine_binding_loss+bag_loss
            # binding_focal_loss=focal_loss(forecast_site_prob, inputs_labels.float())
            # binding_focal_loss=focal_loss(prob, site.float())
            auc_total+=train_auc
            aupr_total+=train_aupr

            auc_total_combine+=train_auc_combine
            aupr_total_combine+=train_aupr_combine

            #保留了四个分类结果
            inx=0
            while inx<K_sub:
                auc_total_bag[inx]+=result_auc_bag[inx]
                aupr_total_bag[inx]+=result_aupr_bag[inx]
                inx+=1

            #训练过程中
            """
            loss需要进行修改，不仅要考虑forecast和target，还要考虑预测结合位点和实际结合位点的关系（结合位点的损失不区分输入和目标，而是一起考虑）；
            """
            #第一个损失函数为时间序列上的预测
            # loss = forecast_loss(forecast, target)
            #结合位点的预测损失,使用交叉熵损失
            # weight = class_weight[inputs_site]
            # all_loss=(loss+binding_loss)/2

            # print('reconstuction_loss '+str(reconstuction_loss)+' '+'train_auc '+str(train_auc))
            print('epoch %d,reconstuction_loss %.4f, train_auc %.4f, train_aupr %.4f  '
                  % (epoch + 1, combine_binding_loss,train_auc,train_aupr))
            cnt += 1

            # loss.backward()
            model.zero_grad()

            # binding_focal_loss.requires_grad_()
            # binding_loss.backward()
            # binding_loss.backward()
            # combine_binding_loss.backward()
            all_loss.backward()
            # reconstuction_loss.backward()
            # all_loss.backward()
            my_optim.step()
            # loss_total += float(loss)
            # loss_total += float(binding_focal_loss)
            loss_total += float(combine_binding_loss)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} |train_auc_combine {:5.4f}| train_aupr_combine {:5.4f}| train_auc {:5.4f}| train_aupr {:5.4f}'.format(epoch+1, (
                time.time() - epoch_start_time), loss_total / cnt,auc_total_combine/cnt,aupr_total_combine/cnt,auc_total/cnt,aupr_total/cnt))
        print('| end of epoch {:3d} | time: {:5.2f}s | train_auc1 {:5.4f} | train_aupr1 {:5.4f}|train_auc2 {:5.4f} | train_aupr2 {:5.4f}|train_auc3 {:5.4f} | train_aupr3 {:5.4f}|train_auc4 {:5.4f} | train_aupr4 {:5.4f}| '.format(epoch+1, (
                time.time() - epoch_start_time), auc_total_bag[0]/cnt, aupr_total_bag[0]/cnt, auc_total_bag[1]/cnt, aupr_total_bag[1]/cnt, auc_total_bag[2]/cnt, aupr_total_bag[2]/cnt, auc_total_bag[3]/cnt, aupr_total_bag[3]/cnt))
        # save_model(model, result_file, epoch)

        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            # performance_metrics = validate(model, valid_loader, args.device,
            #              node_cnt, args.window_size, args.horizon,
            #              result_file=result_file)
            result,Real_prediction,Real_prediction_prob,validate_feature,validate_bag_feature_new,\
            All_bag_result,All_bag_rp,All_bag_rpp=validate_inference_binding_site(model, valid_loader,K_sub)
            MCC = result[0]
            auc = result[1]
            aupr=result[2]
            F1 = result[3]
            Acc = result[4]
            Sen = result[5]
            Spec = result[6]
            Prec = result[7]
            print('validate_MCC: '+str(round(MCC,4))+' '+' validate_auc: '+str(round(auc,4))+' validate_aupr: '+str(round(aupr,4))+' '+' validate_F1: '+str(round(F1,4))+' '+
                  ' validate_Acc: '+str(round(Acc,4))+' '+' validate_Sen: '+str(round(Sen,4))+' '+' validate_Spec: '+str(round(Spec,4))+' '
                   +' validate_Prec: '+str(round(Prec,4)))
            if Acc >= best_validate_Acc:
                best_validate_Acc = Acc
                best_result=result
                best_Real_Predition=Real_prediction
                best_Real_Predition_Prob=Real_prediction_prob
                best_train_feature=Temp_train_feature
                best_validate_feature=validate_feature
                best_train_weight=Temp_train_weight_score
                is_best_for_now = True
            # save model
            if is_best_for_now:
                save_model(model, result_file, fold)

        # best_train_feature = Temp_train_feature
        # best_validate_feature = validate_feature

        # pd.DataFrame(forecast_feature.cpu().detach().numpy()).to_csv('Train_Test_final_feature'+str(epoch)+'.csv')
    #全局表示的各项指标，真实值_预测值，真实值_预测概率，

    StorFile(best_Real_Predition, './Pre-Encoding/data/'+str(species)+'/Result/Global/Real_Predition'+str(fold)+'.csv')
    StorFile(best_Real_Predition_Prob, './Pre-Encoding/data/'+str(species)+'/Result/Global/Real_Predition_prob'+str(fold)+'.csv')
    StorFile(best_train_feature,'./Pre-Encoding/data/'+str(species)+'/Train_Test_feature/Global/Train_feature' + str(fold) + '.csv')
    StorFile(best_validate_feature,'./Pre-Encoding/data/'+str(species)+'/Train_Test_feature/Global/Validate_feature' + str(fold) + '.csv')
    StorFile(best_train_weight,'./Pre-Encoding/data/'+str(species)+'/Train_Test_feature/Global/Train_weight_score' + str(fold) + '.csv') #局部核苷酸评分


    #存储局部表示的预测结果、训练特征和测试特征、真实值与预测标签、真实值与预测值
    StorFile(All_bag_result, './Pre-Encoding/data/'+str(species)+'/Result/Local/All_result' + str(fold) + '.csv')

    #存储验证集上的局部表示
    i=0
    while i<len(validate_bag_feature_new):
        StorFile(validate_bag_feature_new[i],'./Pre-Encoding/data/'+str(species)+'/Train_Test_feature/Local/test_feature' + str(fold) + str(i) + '.csv')
        i+=1

    print(torch.tensor(Temp_train_bag_feature).shape)
    Temp_train_bag_feature=list(torch.tensor(Temp_train_bag_feature).permute(1,0,2).cpu().detach().numpy())
    # print(len(Temp_train_bag_feature))
    # print(len(Temp_train_bag_feature[0]))

    #存储训练阶段的局部表示
    sss=0
    while sss<len(Temp_train_bag_feature):
        print(sss)
        StorFile(Temp_train_bag_feature[sss],'./Pre-Encoding/data/'+str(species)+'/Train_Test_feature/Local/train_feature' + str(fold)+'_' + str(sss)+'.csv')
        sss+=1

    #存储局部表示的真实标签和预测标签
    i = 0
    while i < len(All_bag_rp):
        StorFile(All_bag_rp[i],
                 './Pre-Encoding/data/'+str(species)+'/Result/Local/Real_Predition' + str(fold) +'_' + str(i) + '.csv')
        i += 1

    #存储局部表示的真实标签和预测概率
    i=0
    while i < len(All_bag_rpp):
        StorFile(All_bag_rpp[i],
                 './Pre-Encoding/data/'+str(species)+'/Result/Local/Real_Predition_prob' + str(fold)+'_' + str(i) + '.csv')
        i += 1

    # print('len(best_train_feature) '+str(len(best_train_feature)))
    # print('len(best_train_feature[0]) '+str(len(best_train_feature[0])))
    # print('len(best_validate_feature) '+str(len(best_validate_feature)))
    # print('len(best_validate_feature[0])'+str(len(best_validate_feature[0])))

    print(
        'best_MCC: ' + str(round(best_result[0], 4)) + ' ' + ' best_auc: ' + str(round(best_result[1], 4)) + ' best_aupr: ' + str(
            round(best_result[2], 4)) + ' ' + ' best_F1: ' + str(round(best_result[3], 4)) + ' ' +
        ' best_Acc: ' + str(round(best_result[4], 4)) + ' ' + ' best_Sen: ' + str(
            round(best_result[5], 4)) + ' ' + ' best_Spec: ' + str(round(best_result[6], 4)) + ' '
        + ' best_Prec: ' + str(round(best_result[7], 4)))

    return best_result

import pandas as pd

def test(test_data, args, result_train_file, species, fold):  #
    print(result_train_file)
    model = load_model(result_train_file, fold)

    test_set = ForecastDataset(test_data)
    valid_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)

    result, Real_prediction, Real_prediction_prob, validate_feature, validate_bag_feature_new, \
    All_bag_result, All_bag_rp, All_bag_rpp = validate_inference_binding_site(model, valid_loader, 4)

    MCC = result[0]
    auc = result[1]
    aupr = result[2]
    F1 = result[3]
    Acc = result[4]
    Sen = result[5]
    Spec = result[6]
    Prec = result[7]

    print(
        'validate_MCC: ' + str(round(MCC, 4)) + ' ' + ' validate_auc: ' + str(round(auc, 4)) + ' validate_aupr: ' + str(
            round(aupr, 4)) + ' ' + ' validate_F1: ' + str(round(F1, 4)) + ' ' +
        ' validate_Acc: ' + str(round(Acc, 4)) + ' ' + ' validate_Sen: ' + str(
            round(Sen, 4)) + ' ' + ' validate_Spec: ' + str(round(Spec, 4)) + ' '
        + ' validate_Prec: ' + str(round(Prec, 4)))

    # np.save('./Case_CS_CT/CS/' + str(species) + '/Train_Test_Feature/valid_x1_feature' + str(fold) + '.npy',
    #         valid_x1_feature)
    # np.save('./Case_CS_CT/CS/' + str(species) + '/Train_Test_Feature/valid_x2_feature' + str(fold) + '.npy',
    #         valid_x2_feature)
    StorFile(validate_feature,'./Case_CS_CT/CS/' + str(species) + '/Train_Test_Feature/valid_feature' + str(fold) + '.csv')

    pd.DataFrame(result).to_csv('./Case_CS_CT/CS/' + str(species) + '/Result/result' + str(fold) + '.csv')
    StorFile(Real_prediction, './Case_CS_CT/CS/' + str(species) + '/Result/Real_Prediction' + str(fold) + '.csv')
    StorFile(Real_prediction_prob,
             './Case_CS_CT/CS/' + str(species) + '/Result/Real_Prediction_prob' + str(fold) + '.csv')

    # pd.DataFrame(result).to_csv('./Case_CS_CT/CS/' + str(species) + '/Result/result' + str(fold) + '.csv')
    # StorFile(Real_prediction, './Case_CS_CT/CS/' + str(species) + '/Result/Real_Prediction' + str(fold) + '.csv')
    # StorFile(Real_prediction_prob,
    #          './Case_CS_CT/CS/' + str(species) + '/Result/Real_Prediction_prob' + str(fold) + '.csv')

    # np.save('./Case_CS_CT/CT/' + str(species) + '/Train_Test_Feature/valid_x1_feature' + str(fold) + '.npy',
    #         valid_x1_feature)
    # np.save('./Case_CS_CT/CT/' + str(species) + '/Train_Test_Feature/valid_x2_feature' + str(fold) + '.npy',
    #         valid_x2_feature)
    # np.save('./Case_CS_CT/CT/' + str(species) + '/Train_Test_Feature/valid_feature' + str(fold) + '.npy',
    #         validate_features)
    #
    # StorFile(valid_x1_feature,'./Case_CS_CT/CT/' + str(species) + '/Train_Test_Feature/valid_x1_feature' + str(fold) + '.csv')
    # StorFile(valid_x2_feature,'./Case_CS_CT/CT/' + str(species) + '/Train_Test_Feature/valid_x2_feature' + str(fold) + '.csv')
    # StorFile(validate_feature,'./Case_CS_CT/CT/' + str(species) + '/Train_Test_Feature/valid_feature' + str(fold) + '.csv')
    #
    # pd.DataFrame(result).to_csv('./Case_CS_CT/CT/' + str(species) + '/Result/result' + str(fold) + '.csv')
    # StorFile(Real_prediction, './Case_CS_CT/CT/' + str(species) + '/Result/Real_Prediction' + str(fold) + '.csv')
    # StorFile(Real_prediction_prob,
    #          './Case_CS_CT/CT/' + str(species) + '/Result/Real_Prediction_prob' + str(fold) + '.csv')

    # import pandas as pd
    # pd.DataFrame(result).to_csv('./Motifis_analysis/Score_motif/Mouse_kidney/result0.csv')
    # StorFile(Real_prediction, './Motifis_analysis/Score_motif/Mouse_kidney/Real_Prediction0.csv')
    # StorFile(Real_prediction_prob, './Motifis_analysis/Score_motif/Mouse_kidney/Real_Prediction_prob0.csv')

    # import pandas as pd
    # pd.DataFrame(result).to_csv('./Case_CS_CT/CS/R_R/result0.csv')
    # StorFile(Real_Prediction, './Case_CS_CT/CS/R_R/Real_Prediction0.csv')
    # StorFile(Real_Prediction_Prob, './Case_CS_CT/CS/R_R/Real_Prediction_prob0.csv')
    # StorFile(feature, './Case_CS_CT/CS/R_R/feature0.csv')
    # StorFile(new_motifs, './Case_CS_CT/CS/M_R/motif.csv')
    # StorFile(x_type, './Case_CS_CT/CS/M_R/x_type.csv')
