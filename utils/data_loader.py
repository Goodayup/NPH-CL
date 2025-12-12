import numpy as np
import scipy.sparse as sp

from collections import defaultdict
import warnings

# =========================================================
# START MODIFICATION (解决导入和默认 read_cf 定义)
# =========================================================
warnings.filterwarnings('ignore')

n_genes = 0
n_drugs = 0
dataset = ''
train_gene_set = defaultdict(list)
train_drug_set = defaultdict(list)
test_gene_set = defaultdict(list)
valid_gene_set = defaultdict(list)


def read_cf(file_name):
    """
    默认的协同过滤数据读取函数，用于非 'dgidb5' 数据集 (如 'aminer')。
    假设文件格式是每行一对关联：gene_id <separator> drug_id
    """
    cf = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            
            # 尝试使用空格分隔，如果失败则尝试逗号或制表符
            try:
                items = line.split()
                if len(items) != 2:
                    raise ValueError
            except ValueError:
                try:
                    items = line.split(',')
                    if len(items) != 2:
                        raise ValueError
                except ValueError:
                    items = line.split('\t')
                    if len(items) != 2:
                        raise ValueError
            
            try:
                gene_id = int(items[0])
                drug_id = int(items[1])
                cf.append((gene_id, drug_id))
            except Exception as e:
                # 这一步是为了防止程序因格式错误崩溃，但最好的做法是确保数据干净
                print(f"Warning: Skipping line due to format error or non-integer ID: {line}")
                continue
    
    return np.array(cf, dtype=np.int32)  


def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        # 注意: yelp2018 格式可能是: g_id pos_id1 pos_id2...
        inters = [int(i) for i in tmps.split(" ")] 
        g_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for d_id in pos_ids:
            inter_mat.append([g_id, d_id])
    return np.array(inter_mat)

# =========================================================
# END MODIFICATION (其余代码保持不变)
# =========================================================


def statistics(train_data, valid_data, test_data):
    global n_genes, n_drugs, dataset
    
    # 确保数据集不是空集，防止 max 报错
    if train_data.size == 0 and valid_data.size == 0 and test_data.size == 0:
        print("Warning: All data sets are empty.")
        return

    # 为避免传入空数组时的 max 错误，这里需要额外的检查
    max_gene = 0
    if train_data.size > 0: max_gene = max(max_gene, np.max(train_data[:, 0]))
    if valid_data.size > 0: max_gene = max(max_gene, np.max(valid_data[:, 0]))
    if test_data.size > 0: max_gene = max(max_gene, np.max(test_data[:, 0]))
    
    max_drug = 0
    if train_data.size > 0: max_drug = max(max_drug, np.max(train_data[:, 1]))
    if valid_data.size > 0: max_drug = max(max_drug, np.max(valid_data[:, 1]))
    if test_data.size > 0: max_drug = max(max_drug, np.max(test_data[:, 1]))
    
    n_genes = max_gene + 1
    n_drugs = max_drug + 1

    if dataset not in ['dgidb5', 'GO','Drugbank']:
        n_drugs -= n_genes
        # remap [n_genes, n_genes+n_drugs] to [0, n_drugs]
        train_data[:, 1] -= n_genes
        valid_data[:, 1] -= n_genes
        test_data[:, 1] -= n_genes

    cnt_train, cnt_test, cnt_valid = 0, 0, 0
    for g_id, d_id in train_data:
        train_gene_set[int(g_id)].append(int(d_id))
        train_drug_set[int(d_id)].append(int(g_id))
        cnt_train += 1
    for g_id, d_id in test_data:
        test_gene_set[int(g_id)].append(int(d_id))
        cnt_test += 1
    for g_id, d_id in valid_data:
        valid_gene_set[int(g_id)].append(int(d_id))
        cnt_valid += 1
    print('n_genes: ', n_genes, '\tn_drugs: ', n_drugs)
    print('n_train: ', cnt_train, '\tn_test: ', cnt_test, '\tn_valid: ', cnt_valid)
    print('n_inters: ', cnt_train + cnt_test + cnt_valid)


def build_sparse_graph(data_cf):
    def _bd_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bd_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bd_lap.tocoo()

    def _sd_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_genes  # [0, n_drugs) -> [n_genes, n_genes+n_drugs)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_genes+n_drugs)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_genes + n_drugs, n_genes + n_drugs))
    return _bd_norm_lap(mat), np.array(mat.sum(1)), np.array(mat.sum(0))


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    
    # directory 现在将正确包含 args.data_path 和 dataset
    directory = args.data_path + dataset + '/' 

    print('reading train and test user-item set ...')
    
    # 确定读取函数
    read_fn = globals()['read_cf']
    if dataset == 'dgidb5' :
        # 如果是 dgidb5，使用特定的读取函数
        read_fn = read_cf_yelp2018
    elif dataset=='GO':
        read_fn = read_cf_yelp2018
    elif dataset=='Drugbank':
        read_fn = read_cf_yelp2018
    
    # 构造路径: directory = args.data_path + dataset + '/'
    train_path = directory + 'train.txt'
    test_path = directory + 'test.txt'

    # 核心赋值操作：确保 train_cf 和 test_cf 始终被赋值
    train_cf = read_fn(train_path)
    test_cf = read_fn(test_path)
        
    valid_cf = test_cf
    
    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat, indeg, outdeg = build_sparse_graph(train_cf)

    n_params = {
        'n_genes': int(n_genes),
        'n_drugs': int(n_drugs),
    }
    gene_dict = {
        'train_drug_set': train_drug_set,
        'train_gene_set': train_gene_set,
        'valid_gene_set': None,
        'test_gene_set': test_gene_set,
    }
    print('loading over ...')
    return train_cf, gene_dict, n_params, norm_mat, indeg, outdeg