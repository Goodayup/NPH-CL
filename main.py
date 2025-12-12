import os
import random
import sys # 引入 sys 用于退出
import torch
import numpy as np
from time import time


from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test

import optuna
import joblib
import datetime
from sklearn.metrics import roc_auc_score,average_precision_score


n_genes = 0
n_drugs = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1, K=1, n_drugs=0):
    def sampling(gene_item, train_set, n):
        neg_items = []
        for user, _ in gene_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_drugs))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['genes'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    # 注意: n_drugs 在这里应该使用传入的参数 n_drugs，而不是全局变量
    # 假设这里能够正确访问到 n_drugs
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                     gene_dict['train_gene_set'],
                                                     n_negs * K)).to(device)
    return feed_dict


def opt_objective(trial, args, train_cf, gene_dict, n_params, norm_mat, deg, outdeg):
    valid_res_list = []

    # args.dim = trial.suggest_int('dim', 16, 512)
    args.l2 = trial.suggest_float('l2', 0, 1)
    args.context_hops = trial.suggest_int('context_hops', 1, 6)

    for seed in range(args.runs):
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(args)
        valid_best_result = main(args, seed, train_cf, gene_dict, n_params, norm_mat, deg, outdeg)
        valid_res_list.append(valid_best_result)
    return np.mean(valid_res_list)


def main(args, run, train_cf, gene_dict, n_params, norm_mat, deg, outdeg):
    """define model"""
    # ====== START: DYNAMIC MODEL LOADING MODIFICATION (基于环境变量) ======
    GNN_MODEL = os.environ.get('GNN_MODEL', 'HeatKernel') # 默认使用 HeatKernel
    print(f"Loading GNN Model: {GNN_MODEL}")

    try:
        if GNN_MODEL == 'HeaKernel':
            from model import HeatKernel
            model = HeatKernel(n_params, args, norm_mat, deg).to(device)
                else:
            print(f"Error: Unknown GNN_MODEL '{GNN_MODEL}' specified via environment variable.")
            sys.exit(1) # 退出程序

    except ImportError as e:
        print(f"Error importing model {GNN_MODEL}: {e}. Make sure the class exists in model.py")
        sys.exit(1) # 退出程序

    # ====== END: DYNAMIC MODEL LOADING MODIFICATION ======


    """define optimizer"""
    optimizer = torch.optim.Adam([{'params': model.parameters(),
                                   'lr': args.lr}])
    n_drugs = n_params['n_drugs']

    # ====== START: EVALUATION VARIABLES MODIFICATION ======
        best_results = {
        'auc': 0.0,
        'aupr': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'f1-score': 0.0
    }

    print("start training ...")

    hyper = {"dim": args.dim, "l2": args.l2, "hops": args.context_hops}
    print("Start hyper parameters: ", hyper)

    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        total_loss, mf_loss_sum, emb_loss_sum, cl_loss_sum = 0, 0, 0, 0 # 初始化所有损失的总和
        s = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  gene_dict['train_gene_set'],
                                  s, s + args.batch_size,
                                  args.n_negs,
                                  args.K,
                                  n_drugs)


            optimizer.zero_grad()
            batch_total_loss.backward() # 使用 total_loss 进行反向传播
            optimizer.step()

            total_loss += batch_total_loss.item()
            mf_loss_sum += batch_mf_loss.item()
            emb_loss_sum += batch_emb_loss.item()
            cl_loss_sum += batch_cl_loss.item()

            s += args.batch_size
       

        train_e_t = time()

        # ====== START: LOGGING MODIFICATION ======
        # 打印详细的损失项，包括对比损失
        print(f'Epoch {epoch+1}/{args.epoch} - Total Loss: {round(total_loss, 4)} '
              f'(MF: {round(mf_loss_sum, 4)}, CL: {round(cl_loss_sum, 4)}) '
              f'Time: {round(train_e_t - train_s_t, 2)} s')
        # ====== END: LOGGING MODIFICATION ======


        # ====== START: EVALUATION LOGIC ADDITION ======
        # 每隔一定步数进行一次评估（例如每10个epoch）
        if (epoch + 1) % 10 == 0 or epoch == args.epoch - 1:
            print(f"--- Testing at Epoch {epoch+1} ---")

            # 调用 utils.evaluate.test 函数
            results = test(model, args, gene_dict, n_params, device)

            # 使用 AUC 作为主要评价指标，因为它是性能衡量的标准之一
            current_auc = results.get('auc', 0.0)

            # 检查是否是当前最佳模型
            if current_auc > best_auc:
                best_auc = current_auc
                best_results.update(results) # 更新所有指标
                print(">>> New BEST result found! <<<")

                # 可选：保存最佳模型权重
                # torch.save(model.state_dict(), f'{args.out_dir}best_model_{GNN_MODEL}.pth')

            print(f"Current AUC: {current_auc:.4f}, Best AUC: {best_auc:.4f}")
            print(f"Current AUPR: {results.get('aupr', 0.0):.4f}, Current F1-score: {results.get('f1-score', 0.0):.4f}")
        # ====== END: EVALUATION LOGIC ADDITION ======

    print("End hyper parameters: ", hyper)
    print("--------------------------------------------------")
    print(f"Final BEST Results for {GNN_MODEL}:")

    # 返回最佳 AUC，用于 Optuna 报告
    return best_auc


if __name__ == '__main__':
    """read args"""
    global args, device
    args = parse_args()
    s = datetime.datetime.now()
    print("time of start: ", s)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    """build dataset"""

    # 获取原始数据参数
    train_cf, gene_dict, n_params_raw, norm_mat, deg, outdeg = load_data(args)

    # ====== START: KEY-NAME FIX MODIFICATION (Keep the final structure) ======
    n_params = {
        'n_genes': n_params_raw.get('n_genes') or n_params_raw.get('N_GENES') or n_params_raw.get('n_gene'),
        'n_drugs': n_params_raw.get('n_drugs') or n_params_raw.get('N_DRUGS') or n_params_raw.get('n_drug'),
    }

    if n_params['n_drugs'] is None:
        raise KeyError("Could not find 'n_drugs' or similar key in data_loader output. Check returned dictionary structure.")
    # ====== END: KEY-NAME FIX MODIFICATION ======

    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    trials = 1
    search_space = {'dim': [512], 'context_hops': [2], 'l2': [1e-3]}
    print("search_space: ", search_space)
    print("trials: ", trials)

    # 打印 n_params 确认结构正确
    print(f"Final n_params structure: {n_params}")

    # 将修正后的 n_params 传递给 study.optimize
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: opt_objective(trial, args, train_cf, gene_dict, n_params, norm_mat, deg, outdeg), n_trials=trials)

    e = datetime.datetime.now()
    print(study.best_trial.params)
    print("time of end: ", e)