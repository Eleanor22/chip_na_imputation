import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_preprocessing(data, flag=False):
    """Initial data preprocessing function.

    Args:
        data (pd.DataFrame): data to be preprocessed
    """
    if 'jia_cat' in data.columns:
        data = data[data['jia_cat'] != 'Systemic arthritis']   

    # data.drop(columns=['sacroiliac_tenderness_month.6']) # Dropping 'sacroiliac_tenderness_month.6 because of too many NA values
    # print(f"Unique values in column 'active_sacroilitis_month.0': {data['active_sacroilitis_month.0'].unique()}")
    for col in ['active_sacroilitis_month.0', 'active_sacroilitis_month.6']:
        data.loc[:, col] = data[col].replace('None', 'No')
    print(f"Unique values in column 'active_sacroilitis_month.0': {data['active_sacroilitis_month.0'].unique()}")

    drop_cols = ['visitDate', 'dob', 'zip', 'startDate', 'dist_date', 'sacroiliac_tenderness_month.6']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    # List of object columns to encode
    joint_cols = data.columns[data.columns.str.contains('totjoint')].tolist()
    combined_cols = joint_cols 

    bool_cols = data.columns[data.columns.str.contains('damage|uveitis_new|active_sacroilitis|active_enthesitis|psoriasis|newmed')].tolist()
    # categorical_cols = ['male', 'jia_cat', 'insurance', 'income_cat', 'educate_cat', 'race_amerind', 'race_asian', 
    # 'race_black', 'race_mideast', 'race_nathwn', 'race_white', 'hispanic', 'hlab27', 'anticcp', 'famhx_rheum', 
    # 'famhx_ibd', 'famhx_uveitis', 'famhx_other_aid', 'ibd', 'asthma', 'other_aid', 'iei']
    categorical_cols = ['jia_cat', 'insurance', 'hlab27', 'anticcp']

    categorical_cols.extend(combined_cols)
    categorical_cols.extend(bool_cols)
    
    # for col in ['educate_cat', 'income_cat']:
    #     data[col] = data[col].replace(['Unknown'], np.nan)
    #     data[col] = pd.to_numeric(data[col], errors='coerce')
        
    for col in categorical_cols:
        
        data[col] = data[col].replace(['Unknown'], np.nan)
        # data[col] = data[col].replace(['None'], 'No')
    
    if flag:
        # Create a dictionary to store the encoders
        encoders = {}

        for col in categorical_cols: 
            if col in ['hlab27', 'anticcp']:
                data[col] = pd.Categorical(data[col], categories=['Negative', 'Positive']).codes
                data[col] = data[col].replace(-1, np.nan)  # '-1' for missing in Categorical
                continue

            le = LabelEncoder()
            # Temporarily fill NaNs with a string so LabelEncoder can handle them
            data[col] = data[col].fillna('Missing').astype(str)
            data[col] = le.fit_transform(data[col])
            encoders[col] = le

            if 'Missing' in le.classes_:
                missing_code = le.transform(['Missing'])[0]
                data[col] = data[col].replace(missing_code, np.nan)
                # print(f"Column: {col}, Unique values after replacing: {data[col].unique()}")
                
    return data


def mapping_jadas(data):
    """
    Maps JADAS scores based on the provided columns.
    """
    
    jadas_cols = [col for col in data.columns if col.startswith('jadas_month.')]
    for j_col in jadas_cols:
        time_suffix = j_col.split('.')[-1]

        phys_col = f'PHYSCORE_month.{time_suffix}'
        rheu_col = f'RHEURATE_month.{time_suffix}'
        actj_col = f'ACTJONUM2_month.{time_suffix}'

        if all(col in data.columns for col in [phys_col, rheu_col, actj_col]):
            # print(f"Calculating {j_col} using {phys_col}, {rheu_col}, {actj_col}")
            # Convert to numeric
            data[j_col] = (
                pd.to_numeric(data[phys_col], errors="coerce") +
                pd.to_numeric(data[rheu_col], errors="coerce") +
                pd.to_numeric(data[actj_col], errors="coerce")
            )
            nan_rows = data[j_col].isnull().sum()
            if nan_rows:
                print(f"{j_col} has {nan_rows} NaNs after sum (due to NaNs in components)")
        else:
            print(f"One or more components missing for {j_col}, skipping.")
    return data