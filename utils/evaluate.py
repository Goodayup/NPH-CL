from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score # 导入所需指标

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag
gene_recall_result = dict()
deg_recall = dict()
deg_recall_mean = dict()


# 确保这个全局变量在模块加载时被正确定义，或者从外部传入
# 如果在 main.py 中已经传入，则无需在此处 global
global n_genes, n_drugs, train_gene_set, test_gene_set 


def test(model, args, gene_dict, n_params, device, mode='test'): # 修正函数签名以匹配 main.py 调用

    # 局部变量，从 n_params 中获取。KeyError 已在 main.py 中处理
    n_drugs = n_params['n_drugs']
    n_genes = n_params['n_genes']

    # 获取数据集
    train_gene_set = gene_dict['train_gene_set']
    if mode == 'test':
        test_gene_set = gene_dict['test_gene_set']
    else:
        test_gene_set = gene_dict['valid_gene_set']
        if test_gene_set is None:
            test_gene_set = gene_dict['test_gene_set']


    g_batch_size = args.test_batch_size # 使用传入的 args 中的 test_batch_size
    d_batch_size = args.test_batch_size

    test_genes = list(test_gene_set.keys())
    n_test_genes = len(test_genes)
    n_gene_batchs = n_test_genes // g_batch_size + 1

    genes_gcn_emb, drug_gcn_emb = model.generate()
    
    # 注意: model.rating(genes_gcn_emb, drug_gcn_emb) 应该是全矩阵得分，不是采样得分
    # pred_score = model.rating(genes_gcn_emb,drug_gcn_emb).detach().cpu().numpy().tolist() # 这一行不需要
    
    y_true = []
    y_pred = []
    
    # ----------------------------------------------------
    # 核心：执行采样评估
    # ----------------------------------------------------
    for g_batch_id in range(n_gene_batchs):
        start = g_batch_id * g_batch_size
        end = (g_batch_id + 1) * g_batch_size

        gene_list_batch = test_genes[start:end]
        gene_batch = torch.LongTensor(np.array(gene_list_batch)).to(device)
        g_g_embeddings = genes_gcn_emb[gene_batch]

        # 计算得分矩阵 (Rate_batch 包含所有药物的得分)
        if args.batch_test_flag: # 注意：这里应使用 args.batch_test_flag
             # batch-item test
            n_drug_batchs = n_drugs // d_batch_size + 1
            rate_batch = np.zeros(shape=(len(gene_batch), n_drugs))

            d_count = 0
            for d_batch_id in range(n_drug_batchs):
                d_start = d_batch_id * d_batch_size
                d_end = min((d_batch_id + 1) * d_batch_size, n_drugs)

                drug_batch = torch.LongTensor(np.array(range(d_start, d_end))).view(d_end-d_start).to(device)
                d_g_embddings = drug_gcn_emb[drug_batch]

                d_rate_batch = model.rating(g_g_embeddings, d_g_embddings).detach().cpu().numpy() # 转换为 numpy

                rate_batch[:, d_start:d_end] = d_rate_batch
                d_count += d_rate_batch.shape[1]

            assert d_count == n_drugs
        else:
            # all-item test
            drug_batch = torch.LongTensor(np.array(range(0, n_drugs))).view(n_drugs).to(device)
            d_g_embddings = drug_gcn_emb[drug_batch]
            rate_batch = model.rating(g_g_embeddings, d_g_embddings).detach().cpu().numpy() # 转换为 numpy

        # 采样正负样本并收集用于 AUC/AUPR/F1 的列表
        for i in range(len(gene_list_batch)):
            uid = gene_list_batch[i]
            drug_scores = rate_batch[i]
            pos = test_gene_set[uid]
            train_drug_ass = train_gene_set[uid]
            
            # 排除所有训练和测试关联，找到所有可能的负样本
            all_drugs = set(range(n_drugs))
            all_associated = set(pos) | set(train_drug_ass)
            diff = list(all_drugs - all_associated)
            
            # 随机采样等量的负样本
            random.shuffle(diff)
            neg = diff[0:len(pos)]
            
            # 收集 y_true 和 y_pred
            y_true += [1] * len(pos)
            y_true += [0] * len(neg) # 仅使用采样的负样本
            
            for item in pos:
                y_pred.append(drug_scores[item])
            for item in neg:
                y_pred.append(drug_scores[item])
    
    # ----------------------------------------------------
    # 指标计算和返回 (修正返回值格式)
    # ----------------------------------------------------
    
    # 确保列表非空
    if not y_true:
        return {
            'auc': 0.0, 'aupr': 0.0, 'recall': 0.0, 'precision': 0.0, 'f1-score': 0.0
        }

    # AUC and AUPR (需要原始预测分数)
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    
    # Precision, Recall, F1-score (需要二值化预测)
    # 设定一个阈值（例如 0.5），或者计算 ROC 曲线上的最佳阈值
    # 这里我们使用一个简单的0.5阈值进行二值化
    y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
    
    # 由于是采样评估，这里计算的 F1 等指标准确性不如 AUC/AUPR，但满足输出要求
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    results = {
        'auc': auc,
        'aupr': aupr,
        'recall': recall,
        'precision': precision,
        'f1-score': f1
    }

    return results # 返回字典，与 main.py 逻辑匹配