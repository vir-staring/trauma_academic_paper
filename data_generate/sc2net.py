import pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations_with_replacement

# 配置参数
data_dir = Path("/path/to/data/ss_tck2sc_all/")  # 被试文件夹的父目录
label_file = "/path/to/data/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
output_dir = Path("/path/to/data/scall2net")
output_dir.mkdir(exist_ok=True)

# 加载标签文件，建立INDEX到NETWORK的映射
labels = pd.read_csv(label_file, sep='\t')
idx_to_net = dict(zip(labels['INDEX'], labels['NETWORK']))
networks = sorted(labels['NETWORK'].unique())

# 生成网络对列名（网络内+网络间，间的连接符用下划线）
net_pairs = [f"{a}_{b}" if a != b else a for a, b in combinations_with_replacement(networks, 2)]
print(f"网络数: {len(networks)}, 网络对数: {len(net_pairs)}")

def process_subject(sub_dir, filename, agg_func, metric_suffix):
    """处理单个被试的连接矩阵，聚合为网络级别"""
    data = pd.read_csv(sub_dir / filename, header=None).values
    np.fill_diagonal(data, 0)
    
    # 收集每个网络对的值
    net_values = {pair: [] for pair in net_pairs}
    n = data.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # 只看上三角
            if data[i, j] == 0: continue
            net_i, net_j = idx_to_net.get(i + 1), idx_to_net.get(j + 1)
            if not net_i or not net_j: continue
            # 按字母顺序排序网络名，保证一致性
            pair = "_".join(sorted([net_i, net_j])) if net_i != net_j else net_i
            if pair in net_values:
                net_values[pair].append(data[i, j])
    
    return {f"{pair}_{metric_suffix}": agg_func(vals) if vals else np.nan for pair, vals in net_values.items()}

def process_all_subjects(filename, agg_func, metric_suffix):
    """处理所有被试"""
    results = []
    for sub_dir in sorted(data_dir.glob("sub-*")):
        if (sub_dir / filename).exists():
            row = {'subject': sub_dir.name.replace("sub-", ""), **process_subject(sub_dir, filename, agg_func, metric_suffix)}
            results.append(row)
    return pd.DataFrame(results)

# 处理三种连接指标并保存
metrics = {
    'connetome_meanFA.csv': ('FA', np.mean),
    'connetome_meanMD.csv': ('MD', np.mean),
    'connetome_meanRD.csv': ('RD', np.mean),
    'connetome_meanAD.csv': ('AD', np.mean),
    # 'connetome_meanlength.csv': ('ML', np.mean),
    # 'connetome_streamlines.csv': ('SL', np.sum)
}

for filename, (suffix, agg_func) in metrics.items():
    df = process_all_subjects(filename, agg_func, suffix)
    output_file = output_dir / f"network_{suffix}.csv"
    df.to_csv(output_file, index=False)
    print(f"已保存: {output_file}, 被试数: {len(df)}, 列数: {len(df.columns)}")