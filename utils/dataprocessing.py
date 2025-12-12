import pandas as pd
import numpy as np
import io
import os

# --- 1. 模拟您的输入文件内容 ---
# 假设这是您的疾病列表
disease_list_content = """omimId
"omimId"
"OMIM:P45059"
"OMIM:P19113"
"OMIM:Q9UI32"
"OMIM:P00488"
"OMIM:P35228"
"OMIM:P37059"
"OMIM:P06737"
"OMIM:P11766"
"OMIM:P48728"
"OMIM:P50213"
"OMIM:P30542"
"OMIM:P23219"
"""


# 假设这是您的药物列表 (真实情况应该有164个)
# 这里为了演示只保留部分，但在真实运行时，请确保读入所有的164个
drug_list_content = """drugBankId
"drugBankId"
"DB00303"
"DB00114"
"DB00117"
"DB00142"
"DB01839"
"DB02340"
"DB00125"
"DB00155"
"DB01110"
"DB01234"
"DB01686"
"DB01835"
"DB01997"
"DB02044"
"DB02207"
"DB02234"
"DB02462"
"DB02644"
"DB03100"
"DB03144"
"DB03366"
"DB03449"
"DB03953"
"DB04400"
"DB04534"
"DB05252"
"DB05383"
"DB06879"
"DB06916"
"DB07002"
"DB07003"
"DB07007"
"DB07008"
"DB07011"
"DB07029"
"DB07306"
"DB07318"
"DB07388"
"DB07389"
"DB07405"
"DB08214"
"DB08750"
"DB08814"
"DB09237"
"DB00157"
"DB00783"
"DB13952"
"DB13953"
"DB13954"
"DB13955"
"DB13956"
"DB00114"
"DB00131"
"DB02089"
"DB02320"
"DB02379"
"DB03288"
"DB03496"
"DB03744"
"""

# [已删除] 这里的强制填充代码已被移除
# num_missing_drugs = 1022 - drug_list_content.strip().count('\n') ...

# 药物-疾病关联矩阵 
# ！！！！！！重要：请确保这里的矩阵行数=疾病数，列数=药物数(164) ！！！！！！
# 这里用一个简单的模拟矩阵代替，实际请加载您的完整矩阵
# 假设是一个 13行(疾病) x 50列(药物) 的矩阵
association_matrix_content = """
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""

# --- 2. 加载 ID 列表并创建映射 ---

# 疾病 ID 映射
df_diseases = pd.read_csv(io.StringIO(disease_list_content), header=0)
disease_original_ids = df_diseases['omimId'].str.replace('"', '').tolist()

# 药物 ID 映射
df_drugs = pd.read_csv(io.StringIO(drug_list_content), header=0)
drug_original_ids = df_drugs['drugBankId'].str.replace('"', '').tolist()

print(f"读取到疾病数量: {len(disease_original_ids)}")
print(f"读取到药物数量: {len(drug_original_ids)}")

# --- 3. 加载药物-疾病关联矩阵 ---
# 如果您的实际文件是 .csv (逗号分隔)，请使用 delimiter=','
association_matrix = np.loadtxt(io.StringIO(association_matrix_content), dtype=int)
# 如果是读取真实文件，请使用:
# association_matrix = np.loadtxt('your_matrix_file.txt', dtype=int)

# ！！！！关键检查！！！！
num_diseases_matrix, num_drugs_matrix = association_matrix.shape
print(f"矩阵维度: {num_diseases_matrix} 行 (疾病) x {num_drugs_matrix} 列 (药物)")

if num_diseases_matrix != len(disease_original_ids):
    print(f"❌ 错误：矩阵行数 ({num_diseases_matrix}) 与 疾病列表数量 ({len(disease_original_ids)}) 不一致！")
    # 注意：如果您的矩阵是 药物x疾病 (转置了)，请在这里使用 association_matrix = association_matrix.T 进行转置
else:
    print(f"✅ 疾病数量匹配")

if num_drugs_matrix != len(drug_original_ids):
    print(f"❌ 错误：矩阵列数 ({num_drugs_matrix}) 与 药物列表数量 ({len(drug_original_ids)}) 不一致！")
else:
    print(f"✅ 药物数量匹配")

# --- 4. 构建稀疏关联列表 ---
disease_drug_associations_sparse = {}

for disease_idx in range(num_diseases_matrix):
    # 找出该疾病(行)中，值为1的列索引(药物)
    associated_drug_indices = np.where(association_matrix[disease_idx, :] == 1)[0]
    disease_drug_associations_sparse[disease_idx] = associated_drug_indices.tolist()

# --- 5. 格式化并输出结果 ---
output_lines = []
# 这里按照矩阵的行索引顺序输出，确保 train.txt 的行号对应 matrix 的行号
for disease_id in sorted(disease_drug_associations_sparse.keys()):
    associated_drugs = disease_drug_associations_sparse[disease_id]
    if associated_drugs:
        output_lines.append(f"{disease_id} {' '.join(map(str, associated_drugs))}")
    else:
        output_lines.append(f"{disease_id}")

# -------------------------------------------------------------------------------------------------
# 保存到 train.txt 文件中
output_filepath = "train.txt"
with open(output_filepath, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\n结果已保存到文件：{output_filepath}")
print(f"train.txt 总行数: {len(output_lines)} (应等于疾病数量)")
# -------------------------------------------------------------------------------------------------