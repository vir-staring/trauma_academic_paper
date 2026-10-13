
import pandas as pd


input_file = '/path/to/data/mri_info.tsv'                       # 你的原始TSV文件名
output_file = '/path/to/data/raw/covs/dmri_site_qc.csv'           # 你想保存的新文件名
# 脚本会自动将第一列作为被试编号列，如果你想指定列名，可以像下面这样修改
# --- 列名配置 ---
subject_col = '被试编号'

col1 = 'DMRI_AP_b0_一次质控结果（0/1/2 : 未通过或未找到匹配校正像/通过/计算失败）'
col2 = 'DMRI_PA_人工质控结果（-1/0/1/2 : 不可接受/待质控/可接受/好）'
col3 = '弥散像_计算质量评估结果'


# col1 = 'T1_人工质控结果（-1/0/1/2 : 不可接受/待质控/可接受/好）'
# col2 = 'T2_人工质控结果（-1/0/1/2 : 不可接受/待质控/可接受/好）'
# col3 = '结构像_计算质量评估结果'
final_quality_col = '是否纳入' # 我们将要创建的新列的名称
# ---------------------------------------------------

# 1. 【关键改动】读取TSV文件，并使用 dtype=str 强制所有列都作为字符串类型读入
# 这可以防止 pandas 自动推断类型，从而避免潜在的类型错误
df = pd.read_csv(input_file, sep='\t', dtype=str)

# 2. 【核心删除逻辑】首先，删除'计算质量评估结果'列中为空的行
# 注意：由于所有内容都是字符串，空值(NaN)在这里可能被读作空字符串''
# 为了更稳妥地处理，我们先将空字符串替换为真正的NaN，再删除
df.replace('', pd.NA, inplace=True) # 将空字符串''替换为pandas的空值标记
df.dropna(subset=[col3], inplace=True)

# 3. 【核心编码逻辑】创建三个独立的布尔条件

# smri : T1和T2的结果必须是 '1' 或 '2' 
# dmri : DMRI_AP_b0_一次质控结果 & DMRI_PA_人工质控结果
# 【关键改动】因为我们以字符串形式读取，所以现在要和字符串'1', '2'进行比较
cond_t1 = df[col1].isin(['1', '2']).fillna(False)
cond_t2 = df[col2].isin(['1', '2']).fillna(False)

# 条件3: 计算结果必须是 '好' 或 '可接受' (这部分逻辑不变)
cond_calc = df[col3].isin(['好', '可接受'])

# 4. 将三个条件用 "与" (&) 逻辑合并
final_condition = cond_t1 & cond_t2 & cond_calc

# 5. 根据最终条件创建新列，并将布尔值(True/False)转换为整数(1/0)
df[final_quality_col] = final_condition.astype(int)

# 6. 只保留“被试编号”和新创建的“最终质控结果”列
final_df = df[[subject_col, final_quality_col]]

# 7. 将结果保存为新的TSV文件
final_df.to_csv(output_file, index=False)

print(f"处理完成！")
print(f"原始文件 '{input_file}' 已处理。")
print(f"结果已保存到 '{output_file}'。")