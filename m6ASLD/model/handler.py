import json
from datetime import datetime
import warnings

from data_loader.SiteBinding_dataloader0 import ForecastDataset
from .seq_graphing import Model
# from models.seq_graph import Model

import torch.utils.data as torch_data
import time
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from .Utils import *

from utils.math_utils import evaluate

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

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_PepBindA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_PepBindA.pt')
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

def validate_inference_binding_site(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, ( inputs, inputs_labels, target, target_labels) in enumerate(
            dataloader):
            inputs = inputs  # 输入cgr序列
            target = target  # 目标CGR序列

            inputs_labels = inputs_labels  # 输入结合位点
            target_labels = target_labels  # 目标结合位点
            criterion = torch.nn.BCELoss(reduction='mean')
            # inputs = normalized_input(inputs)
            # print(inputs.shape)  # torch.Size([32, 12, 51])
            # print(inputs_labels.shape)  # 32x12
            forecast_result,forecast_result_bag,x_feature_combine_result,forecast_feature= model(inputs,inputs)
            # print(forecast_result)
            # prediction,label,_,_=Sequence_reduction(inputs_site,forecast_result,train_info)
            # prediction,label,_,_=Sequence_reduction(inputs_site,forecast_result,train_info)
            # print('forecast_result.shape '+str(forecast_result.shape))
            # result,Real_Prediction,Real_Prediction_Prob=Indicator(inputs_labels,forecast_result)
            result,Real_Prediction,Real_Prediction_Prob=Indicator(inputs_labels,x_feature_combine_result)
            combine_binding_loss = criterion(x_feature_combine_result, inputs_labels.float())
            #
            validate_auc, _, _ = auroc(forecast_result, inputs_labels)
            validate_aupr, _, _ = auprc(forecast_result, inputs_labels)
            # print('validate_auc: '+str(validate_auc)+' '+'validate_aupr: '+str(validate_aupr))
            result[2]=round(validate_aupr,4)

            labels_real = list(inputs_labels.contiguous().view(-1).detach().numpy())
            forecast_feature = list(forecast_feature.detach().numpy())
            xx = 0
            while xx < len(forecast_feature):
                forecast_feature[xx]=list(forecast_feature[xx])
                forecast_feature[xx].append(int(labels_real[xx]))
                xx += 1

        # mean_validate_aupr=aupr_total/cnt
    # print('validate_auc0: '+str(mean_validate_auc)+' '+' validate_aupr0: '+str(mean_validate_aupr))
    return  result,Real_Prediction,Real_Prediction_Prob,forecast_feature,combine_binding_loss,validate_auc

def train(train_data, valid_data, args,result_file,fold):
    # node_cnt = int((train_data.shape[1]-3)/3) #100 (固定窗口大小或自适应窗口大小)
    node_cnt = 256
    print('node_cnt '+str(node_cnt))
    model = Model(node_cnt, 1, args.window_size, args.multi_layer, horizon=args.horizon)
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

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon)
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
    Train_Loss=[]
    Train_auc=[]
    Test_loss=[]
    Test_auc=[]
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        auc_total=0
        aupr_total=0

        auc_total_combine = 0
        aupr_total_combine = 0


        auc_total_bag=[0]*4
        aupr_total_bag=[0]*4
        Temp_train_feature=[]
        for i, (
        inputs, inputs_labels,target, target_labels) in enumerate(
                train_loader):
            inputs = inputs  # 输入cgr序列
            target = target  # 目标CGR序列

            inputs_labels = inputs_labels  # 输入结合位点
            target_labels = target_labels  # 目标结合位点
            # print(inputs_labels.shape) #145x1
            # print(inputs.shape) #torch.Size([32, 12, 51])
            # print(inputs_labels.shape) #32x12

            # inputs= normalized_input(inputs)
            # target= normalized_input(target)

            # forecast, _ = model(inputs,input_prob) #32x12x100 input-target
            forecast_site_prob,forecast_site_prob_bag,x_feature_combine_result,forecast_feature= model(inputs,inputs) #32x12x100 结合位点

            # forecast_site_prob_bags=torch.split(forecast_site_prob_bag,4,-1)


            labels_real = list(inputs_labels.contiguous().view(-1).detach().numpy())
            forecast_feature = list(forecast_feature.detach().numpy())
            xx = 0
            while xx < len(forecast_feature):
                forecast_feature[xx]=list(forecast_feature[xx])
                forecast_feature[xx].append(int(labels_real[xx]))
                xx += 1
            Temp_train_feature.extend(forecast_feature)
            # StorFile(forecast_feature, 'mm_Pse_472_feature/Train_Test_final_feature'+'_' + str(epoch)+'_'+str(i) + '.csv')
            # StorFile(forecast_feature,
            #          'mm_m6A_725/Train_Test_final_feature' + '_' + str(epoch) + '_' + str(i) + '.csv')
            # pd.DataFrame(forecast_feature.detach().numpy()).to_csv('Train_Test_final_featurex' + str(epoch) + '.csv',header=None,index=None)
            #此处增加mask，即重新调整
            # scale=0.5
            # _,prob,site=Index_Mask(forecast_site_prob,inputs_site)

            # forecast_site_label = Prediction_label(forecast_site_prob).contiguous()
            train_auc,_,_=auroc(forecast_site_prob,inputs_labels)
            train_aupr,_,_=auprc(forecast_site_prob,inputs_labels)

            train_auc_combine,_,_=auroc(x_feature_combine_result,inputs_labels)
            train_aupr_combine,_,_=auprc(x_feature_combine_result,inputs_labels)

            bags = torch.chunk(forecast_site_prob_bag, 4, dim=1)
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


            # print('inputs_labels.shape '+str(inputs_labels.shape))
            # print('forecast_site_prob.shape '+str(forecast_site_prob.shape))
            # train_auc,_,_=auroc(prob,site)
            # train_aupr,_,_=auprc(prob,site)
            # reconstuction_loss = forecast_loss(forecast, inputs)
            binding_loss = criterion(forecast_site_prob, inputs_labels.float())
            combine_binding_loss = criterion(x_feature_combine_result, inputs_labels.float())


            bag_loss=0
            j=0
            while j<len(bags):
                bag_loss+=criterion(bags[j], inputs_labels.float())
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
            while inx<4:
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
        Train_Loss.append([(loss_total/cnt)])
        Train_auc.append([auc_total_combine/cnt])

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} |train_auc_combine {:5.4f}| train_aupr_combine {:5.4f}| train_auc {:5.4f}| train_aupr {:5.4f}'.format(epoch+1, (
                time.time() - epoch_start_time), loss_total / cnt,auc_total_combine/cnt,aupr_total_combine/cnt,auc_total/cnt,aupr_total/cnt))
        print('| end of epoch {:3d} | time: {:5.2f}s | train_auc1 {:5.4f} | train_aupr1 {:5.4f}|train_auc2 {:5.4f} | train_aupr2 {:5.4f}|train_auc3 {:5.4f} | train_aupr3 {:5.4f}|train_auc4 {:5.4f} | train_aupr4 {:5.4f}| '.format(epoch+1, (
                time.time() - epoch_start_time), auc_total_bag[0]/cnt, aupr_total_bag[0]/cnt, auc_total_bag[1]/cnt, aupr_total_bag[1]/cnt, auc_total_bag[2]/cnt, aupr_total_bag[2]/cnt, auc_total_bag[3]/cnt, aupr_total_bag[3]/cnt))
        save_model(model, result_file, epoch)

        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            # performance_metrics = validate(model, valid_loader, args.device,
            #              node_cnt, args.window_size, args.horizon,
            #              result_file=result_file)
            result,Real_prediction,Real_prediction_prob,validate_feature,losstr,auctr=validate_inference_binding_site(model, valid_loader)
            Test_loss.append([float(losstr)])
            Test_auc.append([auctr])
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
                is_best_for_now = True
            # save model
            # if is_best_for_now:
            #     save_model(model, result_file)
        best_train_feature = Temp_train_feature
        best_validate_feature = validate_feature
        # pd.DataFrame(forecast_feature.detach().numpy()).to_csv('Train_Test_final_feature'+str(epoch)+'.csv')
    # StorFile(best_Real_Predition, './utils/0.RNA_m_process/m6A/M41/better/Real_Predition'+str(fold)+'.csv')
    # StorFile(best_Real_Predition_Prob, './utils/0.RNA_m_process/m6A/M41/better/Real_Predition_prob'+str(fold)+'.csv')
    # StorFile(best_Real_Predition, './utils/Case_study/Real_Predition'+str(fold)+'.csv')
    # StorFile(best_Real_Predition_Prob, './utils/Case_study/Real_Predition_prob'+str(fold)+'.csv')
    # StorFile(Train_Loss, './utils/0.RNA_m_process/m6A/M41_ablation/all/train_loss.csv')
    # StorFile(Train_auc, './utils/0.RNA_m_process/m6A/M41_ablation/all/train_auc.csv')
    # StorFile(Test_loss,
    #          './utils/0.RNA_m_process/m6A/M41_ablation/all/Test_loss.csv')
    # StorFile(Test_auc,
    #          './utils/0.RNA_m_process/m6A/M41_ablation/all/Test_auc.csv')
    # print('len(best_train_feature) '+str(len(best_train_feature)))
    # print('len(best_train_feature[0]) '+str(len(best_train_feature[0])))
    # print('len(best_validate_feature) '+str(len(best_validate_feature)))
    # print('len(best_validate_feature[0])'+str(len(best_validate_feature[0])))

    # StorFile(best_train_feature,'./utils/0.RNA_m_process/m6A/M41/better/Train_Test_feature/best_train_feature'+str(fold)+'.csv')
    # StorFile(best_validate_feature,'./utils/0.RNA_m_process/m6A/M41/better/Train_Test_feature/best_validate_feature'+str(fold)+'.csv')

    print(
        'best_MCC: ' + str(round(best_result[0], 4)) + ' ' + ' best_auc: ' + str(round(best_result[1], 4)) + ' best_aupr: ' + str(
            round(best_result[2], 4)) + ' ' + ' best_F1: ' + str(round(best_result[3], 4)) + ' ' +
        ' best_Acc: ' + str(round(best_result[4], 4)) + ' ' + ' best_Sen: ' + str(
            round(best_result[5], 4)) + ' ' + ' best_Spec: ' + str(round(best_result[6], 4)) + ' '
        + ' best_Prec: ' + str(round(best_result[7], 4)))
    return forecast_feature,best_result


def test(test_data, args, result_train_file, result_test_file): #
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    result,Real_Prediction,Real_Prediction_Prob=validate_inference_binding_site(model, test_loader)
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

