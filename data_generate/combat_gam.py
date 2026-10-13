from neuroHarmonize import harmonizationLearn
import neuroHarmonize as nh
import pandas as pd
import numpy as np
import os


def data_filter(brain_data_path,cov_data_path,qc_data_path):

    cov_data_df = pd.read_csv(cov_data_path,index_col=0)
    brain_data_df = pd.read_csv(brain_data_path,index_col=0)
    qc_data_df = pd.read_csv(qc_data_path,index_col=0)
    
    # drop sub with abnormal brain feature
    nan_rows_brain = brain_data_df.index[brain_data_df.isna().any(axis=1)].tolist()
    print(f'brain feature abnormal:{len(nan_rows_brain)}')
    print(nan_rows_brain)
    # drop sub from small sample site
    nan_rows_cov = cov_data_df.index[cov_data_df.isna().any(axis=1)].tolist()
    print(f'small site removed:{len(nan_rows_cov)}')
    print(nan_rows_cov)
    # drop sub with abnormal func image
    nan_rows_qc = qc_data_df.index[qc_data_df['是否纳入'] == 0].tolist()
    print(f'func img abnormal:{len(nan_rows_qc)}')
    print(nan_rows_qc)
    
    nan_rows = list(set( nan_rows_brain + nan_rows_cov + nan_rows_qc))
    # nan_rows = list(set( nan_rows_brain + nan_rows_cov ))
    print(f'sub removed : {len(nan_rows)}')

    
    cov_data_df = cov_data_df.drop(index=nan_rows)
    brain_data_df = brain_data_df.drop(index=nan_rows)
    # drop sub_id col
    # brain_data : pd2array
    # cov_data : pd
    # cov_data = cov_data_df.drop(columns=['sub_id'])
    brain_data = np.array(brain_data_df)
    # print(brain_data)
    # 检查是否存在非数值类型

    
    return brain_data,brain_data_df,cov_data_df



## input data path
cov_data_path = '/path/to/data/raw/covs/dmri_combat_cov.csv'
qc_data_path = '/path/to/data/raw/covs/dmri_site_qc.csv'
brain_data_path = '/path/to/data/raw/brain/dmri_network_all.csv'

## combat gam model save path
model_save_path = '/path/to/data/processed/combat_model_dmri_siteqc'

## harmonized brain data save path
harmonized_brain_data_path = '/path/to/data/processed/dmri_net_all_harmonized_siteqc.csv'

## data filter
brain_data,brain_data_df,cov_data_df = data_filter(brain_data_path, cov_data_path, qc_data_path)
# print(brain_data.dtype)
# combat gam
combat_model, brain_data_adj = harmonizationLearn(brain_data, cov_data_df,smooth_terms=['Age_months'])

# replace brain feature with harmonized brain data
brain_data_df.iloc[:,:] = brain_data_adj

# save brain feature,model
brain_data_df.to_csv(harmonized_brain_data_path,index=True)

# print(brain_data_adj)
    
if os.path.exists(model_save_path):
    os.remove(model_save_path)
    
nh.saveHarmonizationModel(combat_model, model_save_path)