import os
import random
import sys 
import torch
import numpy as np
from time import time
import datetime
import joblib
import matplotlib
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

# 解决远程服务器无图形界面问题，用于保存图片
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# 引入 seaborn 以获取更好看的调色板
import seaborn as sns

# 假设 model.py, utils.parser, utils.data_loader, utils.evaluate, analyze_clusters 存在
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from analyze_clusters import analyze_clusters 

# --- 全局变量定义 (为保持兼容性) ---
n_genes = 0
n_drugs = 0
# --- 

# ==============================================================================
# 修正后的案例 3 可视化分析函数 (宏观视角 - 论文出版级优化版 - 美观配色去背景)
# ==============================================================================

def visualize_case_study_3(model, n_params, args, device):
    """
    实现案例 3：宏观 T-SNE 可视化 (论文出版级画质 - 美化版)
    
    修改点：
    1. 配色更换为更美观的 Seaborn 'deep' 调色板。
    2. 移除聚类中心标签 (C0-C7) 的白色背景框。
    3. 调整图例布局，确保所有文字完整清晰显示。
    """
    print("\n" + "="*50)
    print(">>> 正在执行案例 3：宏观语义聚类可视化 (T-SNE) - 美化版 <<<")
    print("="*50)
    
    N_GENES = n_params['n_genes']
    N_DRUGS = n_params['n_drugs'] 
    
    # I. 获取最终嵌入
    model.eval()
    with torch.no_grad():
        final_genes_emb, final_drugs_emb = model.generate(split=True)
        
    all_embeddings = torch.cat([final_genes_emb, final_drugs_emb], dim=0).cpu().numpy()
    total_entities_count = all_embeddings.shape[0]

    # ------------------------------------------------------------------
    # II. 降维和 K-Means 聚类
    # ------------------------------------------------------------------
    
    # 1. 归一化嵌入
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(all_embeddings)

    # 2. T-SNE 降维
    print(f"运行 T-SNE 降维 (总计 {total_entities_count} 个实体)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=args.seed, metric='cosine', learning_rate='auto', verbose=0)
    reduced_embeddings = tsne.fit_transform(normalized_embeddings)

    # 3. 运行 K-Means 聚类
    K_CLUSTERS = 8 
    print(f"运行 K-Means 聚类 ({K_CLUSTERS} 个群集)...")
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=args.seed, n_init='auto')
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    # ------------------------------------------------------------------
    # III. 绘制最终论文用图：基于 K-Means 聚类 (美化配色与布局)
    # ------------------------------------------------------------------
    
    print("绘制最终图表：基于 K-Means 聚类 (美化配色，去除标签背景) ...")
    
    # 设置画板大小，保持较大尺寸
    fig, ax = plt.subplots(figsize=(30, 20))
    
    # --- [修改点1] 更换配色方案 ---
    # 使用 seaborn 的 'deep' 调色板，颜色更柔和专业
    # 如果需要更鲜艳的，可以换成 'bright' 或 'Set2'
    palette = sns.color_palette("colorblind", K_CLUSTERS)
    
    # --- 绘图循环 ---
    for cluster_id in range(K_CLUSTERS):
        indices = np.where(cluster_labels == cluster_id)[0]
        color = palette[cluster_id]
        
        # 区分基因和药物的子集
        gene_sub_indices = indices[indices < N_GENES]
        drug_sub_indices = indices[indices >= N_GENES]
        
        # 1. 绘制基因点 (Marker: 圆形 'o')
        ax.scatter(reduced_embeddings[gene_sub_indices, 0], 
                   reduced_embeddings[gene_sub_indices, 1], 
                   label=f'Cluster {cluster_id}', 
                   alpha=0.8,        # 提高不透明度使颜色更实
                   s=200, 
                   marker='o', 
                   edgecolors='white',
                   linewidth=1,
                   color=color)
        
        # 2. 绘制药物点 (Marker: 三角形 '^')
        ax.scatter(reduced_embeddings[drug_sub_indices, 0], 
                   reduced_embeddings[drug_sub_indices, 1], 
                   label='', 
                   alpha=0.8, 
                   s=240,            # 稍微加大三角形大小
                   marker='^', 
                   edgecolors='white',
                   linewidth=1,
                   color=color)
        
        # 标注聚类中心点
        if len(indices) > 20: 
            center_x = np.median(reduced_embeddings[indices, 0])
            center_y = np.median(reduced_embeddings[indices, 1])
            
            # --- [修改点2] 移除文本背景框 ---
            ax.annotate(
                f'C{cluster_id}', 
                (center_x, center_y), 
                textcoords="offset points", 
                xytext=(0, 0), 
                ha='center', 
                va='center',
                fontsize=28,          # 字号加大
                weight='black',       # 最粗体
                color='black'
                # 已移除 bbox 参数
            )

    # --- 装饰图表 ---
    
    # 标题和坐标轴
    ax.set_title(f"Macro-Semantic Clustering Visualization (K={K_CLUSTERS})", fontsize=32, pad=25, weight='bold')
    #ax.set_xlabel("T-SNE Dimension 1", fontsize=30, labelpad=15)
    #ax.set_ylabel("T-SNE Dimension 2", fontsize=30, labelpad=15)
    
    # 坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # --- [修改点3] 优化图例显示 ---
    # 构造自定义图例句柄
    legend_handles_clusters = []
    for i in range(K_CLUSTERS):
        legend_handles_clusters.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=28, label=f'Cluster {i}'))

    legend_handles_types = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=28, label='Gene'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=28, label='Drug')
    ]
    
    # 将图例放置在图像右侧外部，并留出足够空间
    # 调整 bbox_to_anchor 的 x 坐标，确保不被切断
    
    # 图例 1: Semantic Clusters
    first_legend = ax.legend(handles=legend_handles_clusters, 
                             title="Semantic Clusters", 
                             title_fontsize=28,
                             bbox_to_anchor=(1.03, 1.0), 
                             loc='upper left', 
                             ncol=1, 
                             fontsize=28,
                             frameon=False,
                             borderpad=1.0,   # 增加内部边距
                             labelspacing=1.2 # 增加行间距
                            )
    
    ax.add_artist(first_legend)
    
    # 图例 2: Entity Type
    ax.legend(handles=legend_handles_types, 
              title="Entity Type", 
              title_fontsize=22,
              bbox_to_anchor=(1.03, 0.35), # 调整垂直位置
              loc='upper left', 
              ncol=1, 
              fontsize=28,
              frameon=False,
              borderpad=1.0,
              labelspacing=1.2
             )

    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 关键：调整整个布局，在右侧留出足够空间给图例
    plt.subplots_adjust(right=0.82, left=0.08, top=0.92, bottom=0.1)
    
    # 保存文件
    save_file_macro = f"{args.out_dir}case_study_3_macro_beautified_{os.environ.get('GNN_MODEL', 'HeatKernel')}_seed{args.seed}.png"
    # 使用 bbox_inches='tight' 会重新计算边界，可能会抵消 subplots_adjust 的效果
    # 这里我们信任 subplots_adjust 的结果，或者保存时只用 tight_layout
    plt.savefig(save_file_macro, dpi=300) 
    print(f"美化版 T-SNE 聚类可视化已保存到: {save_file_macro}")
    plt.close()
    
    # 返回聚类标签和总实体数
    return cluster_labels, total_entities_count

# ==============================================================================
# 辅助函数 and Core Functions (保持不变)
# ==============================================================================

def get_feed_dict(train_entity_pairs, gene_dict, start, end, n_negs=1, K=1, n_drugs=0, device='cpu'):
    
    def sampling(gene_item, train_set, n, n_drugs_param):
        neg_items = []
        gene_item_cpu = gene_item.cpu().numpy() 
        for user, _ in gene_item_cpu:
            user = int(user)
            negitems = []
            for i in range(n):
                while True:
                    negitem = random.choice(range(n_drugs_param)) 
                    if user in train_set and negitem not in train_set[user]:
                        break
                    elif user not in train_set:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['genes'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    
    train_set = gene_dict['train_gene_set'] 

    feed_dict['neg_items'] = torch.LongTensor(sampling(
        entity_pairs,
        train_set, 
        n_negs * K,
        n_drugs
    )).to(device)
    return feed_dict

def opt_objective(trial, args, train_cf, gene_dict, n_params, norm_mat, deg, outdeg):
    valid_res_list = []
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
        valid_best_auc = main(args, seed, train_cf, gene_dict, n_params, norm_mat, deg, outdeg)
        valid_res_list.append(valid_best_auc)
    return np.mean(valid_res_list)


def main(args, run, train_cf, gene_dict, n_params, norm_mat, deg, outdeg):
    
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() and gpu_id.isdigit() else torch.device("cpu")
    

    
    # ====== START: DYNAMIC MODEL LOADING MODIFICATION (基于环境变量) ======
    GNN_MODEL = os.environ.get('GNN_MODEL', 'HeatKernel')
    print(f"Loading GNN Model: {GNN_MODEL}")
    
    try:
        if GNN_MODEL == 'HeatKernel':
            from model import HeatKernel
            model = HeatKernel(n_params, args, norm_mat, deg).to(device)
        elif GNN_MODEL == 'APPNP':
            from model import APPNP
            model = APPNP(n_params, args, norm_mat, deg).to(device) 
        else:
            print(f"Error: Unknown GNN_MODEL '{GNN_MODEL}' specified via environment variable.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Error importing model {GNN_MODEL}: {e}. Make sure the class exists in model.py")
        sys.exit(1)
        
    # ====== END: DYNAMIC MODEL LOADING MODIFICATION ======

    
    """define optimizer"""
    optimizer = torch.optim.Adam([{'params': model.parameters(),
                                   'lr': args.lr}])
    n_drugs = n_params['n_drugs']
    
    best_auc = 0
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
    
    # 用于保存最佳模型参数的文件路径
    model_save_path = f'{args.out_dir}best_model_{GNN_MODEL}_seed{run}.pth'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        total_loss, mf_loss_sum, emb_loss_sum, cl_loss_sum = 0, 0, 0, 0 
        s = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  gene_dict, 
                                  s, s + args.batch_size,
                                  args.n_negs,
                                  args.K,
                                  n_drugs,
                                  device=device) 
            
            try:
                batch_total_loss, batch_mf_loss, batch_emb_loss, batch_cl_loss = model(batch)
            except ValueError as e:
                print(f"ERROR in model call: {e}")
                print("Check if model.py/HeatKernel.forward returns 4 values: total_loss, mf_loss, emb_loss, cl_loss.")
                sys.exit(1)


            optimizer.zero_grad()
            batch_total_loss.backward() 
            optimizer.step()

            total_loss += batch_total_loss.item()
            mf_loss_sum += batch_mf_loss.item()
            emb_loss_sum += batch_emb_loss.item()
            cl_loss_sum += batch_cl_loss.item()
            
            s += args.batch_size
            
        train_e_t = time()
        
        # ====== START: LOGGING MODIFICATION ======
        print(f'Epoch {epoch+1}/{args.epoch} - Total Loss: {round(total_loss, 4)} '
              f'(MF: {round(mf_loss_sum, 4)}, CL: {round(cl_loss_sum, 4)}) '
              f'Time: {round(train_e_t - train_s_t, 2)} s')
        # ====== END: LOGGING MODIFICATION ======

        
        # ====== START: EVALUATION LOGIC ADDITION ======
        if (epoch + 1) % 10 == 0 or epoch == args.epoch - 1:
            results = test(model, args, gene_dict, n_params, device)
            current_auc = results.get('auc', 0.0) 
            
            if current_auc > best_auc:
                best_auc = current_auc
                best_results.update(results)
                print(">>> New BEST result found! Saving model state. <<<")
                torch.save(model.state_dict(), model_save_path)
                
            print(f"Current AUC: {current_auc:.4f}, Best AUC: {best_auc:.4f}")
            print(f"Current AUPR: {results.get('aupr', 0.0):.4f}, Current F1-score: {results.get('f1-score', 0.0):.4f}")
        # ====== END: EVALUATION LOGIC ADDITION ======

    print("End hyper parameters: ", hyper)
    
    # ====== START: CASE STUDY 3 CALL (Visualizing Best Model) ======
    
    # 1. 重新加载最佳模型权重
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Successfully loaded best model state from {model_save_path}")
    else:
        print("Warning: Best model state not found. Using final epoch weights for visualization.")

    # 2. 调用可视化函数，并接收聚类标签和总实体数
    cluster_labels, total_entities_count = visualize_case_study_3(model, n_params, args, device)
    
    # 3. 调用分析函数，打印出每个聚类中的 ID
    analyze_clusters(cluster_labels, n_params['n_genes'], n_params['n_drugs'], args, total_entities_count)
    
    # ====== END: CASE STUDY 3 CALL ======

    print("--------------------------------------------------")
    print(f"Final BEST Results for {GNN_MODEL}:")
    metrics_to_print = ['auc', 'aupr', 'recall', 'precision', 'f1-score']
    for metric in metrics_to_print:
        print(f"  {metric.upper()}: {best_results.get(metric, 0.0):.4f}")
        
    print("--------------------------------------------------")
    
    return best_auc


# ==============================================================================
# 主执行块
# ==============================================================================

if __name__ == '__main__':
    args = parse_args()
    s = datetime.datetime.now()
    print("time of start: ", s)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    """build dataset"""
    
    train_cf, gene_dict, n_params_raw, norm_mat, deg, outdeg = load_data(args)
    
    # ====== START: KEY-NAME FIX MODIFICATION ======
    n_params = {
        'n_genes': n_params_raw.get('n_genes') or n_params_raw.get('N_GENES') or n_params_raw.get('n_gene'),
        'n_drugs': n_params_raw.get('n_drugs') or n_params_raw.get('N_DRUGS') or n_params_raw.get('n_drug'),
    }
    
    if n_params['n_drugs'] is None or n_params['n_genes'] is None:
        raise KeyError("Could not find required 'n_genes' or 'n_drugs' in data_loader output.")
    # ====== END: KEY-NAME FIX MODIFICATION ======

    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    
    # Optuna setup
    trials = 1
    search_space = {'dim': [args.dim], 'context_hops': [args.context_hops], 'l2': [args.l2]}
    
    if os.environ.get('OPTUNA_STUDY') == '1':
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(lambda trial: opt_objective(trial, args, train_cf, gene_dict, n_params, norm_mat, deg, outdeg), n_trials=trials)
        GNN_MODEL = os.environ.get('GNN_MODEL', 'HeatKernel') 
        joblib.dump(study,
                     f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_study_{GNN_MODEL}.pkl')
        print(study.best_trial.params)
    else:
        print("Running single main process...")
        main(args, args.seed, train_cf, gene_dict, n_params, norm_mat, deg, outdeg)
        
    e = datetime.datetime.now()
    print("time of end: ", e)