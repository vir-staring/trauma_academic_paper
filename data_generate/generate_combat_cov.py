
import pandas as pd
import re

# 读取横断队列被试信息表
df_subj = pd.read_csv('/path/to/data/raw/covs/brain_covs.tsv', sep="\t")

# 转换采集年龄为月数
def age_to_months(age_str):
    match = re.match(r'(\d+)岁(\d+)个月(\d+)天', str(age_str))
    if match:
        years, months, days = map(int, match.groups())
        return years * 12 + months + days / 30
    return None

df_subj['Age_months'] = df_subj['采集年龄'].apply(age_to_months)

# 性别转为01编码（男=1，女=0）
df_subj['Gender'] = df_subj['性别'].map({'男': 1, '女': 0})

# 提取被试编号
df_subj = df_subj.rename(columns={'被试编号': 'sub_id'})

# 读取mri_info，删除采集设备列为空的行
df_mri = pd.read_csv('/path/to/data/mri_info.tsv', sep='\t')
df_mri = df_mri.dropna(subset=['采集设备'])

# 按照横断队列被试信息表中被试编号的顺序排列mri_info
df_mri = df_mri.set_index('被试编号').reindex(df_subj['sub_id']).reset_index()

# 拼接站点和设备
df_mri['SITE'] = df_mri['所属站点'].astype(str) + '-' + df_mri['采集设备'].astype(str)

# 直接组合列（现在行顺序已对齐）
df_result = pd.DataFrame({
    'sub_id': df_subj['sub_id'],
    'SITE': df_mri['SITE'],
    'Gender': df_subj['Gender'],
    'Age_months': df_subj['Age_months']
})

# 保存
df_result.to_csv('/path/to/data/raw/covs/dmri_combat_cov.csv', index=False)
print(f'✓ combat_cov.csv 已生成，共 {len(df_result)} 行')


# 将指定站点的数据置为NaN
sites_to_nan = ['SITE_TO_EXCLUDE_1',
                'SITE_TO_EXCLUDE_2',
                'SITE_TO_EXCLUDE_3',
                ]

if sites_to_nan:
    import numpy as np
    mask = df_result['SITE'].isin(sites_to_nan)
    df_result.loc[mask, ['Gender', 'Age_months']] = np.nan
    print(f'将 {mask.sum()} 个被试的Gender和Age_months置为NaN（来自 {len(sites_to_nan)} 个站点）')
    df_result.to_csv('/path/to/data/raw/covs/dmri_combat_cov.csv', index=False)
    print('combat_cov.csv 已更新')