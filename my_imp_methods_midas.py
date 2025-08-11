import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
from sklearn.model_selection import train_test_split
from missforest import MissForest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import data_preprocessing #import mapping_jadas
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import impyute
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import MIDASpy as md
import miceforest as mf
import os
import gc


def my_missforest_dflt(train_data, test_data, param, output_dir_imp, folder='missforest'):
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])

    categorical = ['male', 'jia_cat', 'insurance', 'income_cat', 'educate_cat',
                'race_amerind', 'race_asian', 'race_black', 'race_mideast', 'race_nathwn', 'race_white', 'hispanic',
                'hlab27', 'anticcp', 'famhx_rheum', 'famhx_ibd', 'famhx_uveitis', 'famhx_other_aid',
                'ibd', 'asthma', 'other_aid', 'iei']
    categorical += ['totjoint_month.0', 'totjoint_month.6',
                'rad_damage_month.0', 
                'uveitis_new_month.0', 'uveitis_new_month.6',
                'active_sacroilitis_month.0', 'active_sacroilitis_month.6',
                'active_enthesitis_month.0', 'active_enthesitis_month.6',
                'psoriasis_month.0', 'psoriasis_month.6',
                'newmed_month.0', 'newmed_month.6']
    categorical = [col for col in categorical if col in train_data.columns]

    column_bounds = {
        'PHYSCORE_month': (0, 10),
        'RHEURATE_month': (0, 10),
        'ACTJONUM2_month': (0, 10),
        'GHA_T_SCORE_month': (20, 80),
        'DIS_ACT_month': (0, 10),
        'RHEUM_PAIN_month': (0, 10)
        # 'diff_dx_symp_mth': (0, None)
    }
    
    mf = MissForest(early_stopping=False, categorical=categorical)
    train_imputed_data = mf.fit_transform(train_data)
    test_imputed_data= mf.transform(test_data)

    for df in [train_imputed_data, test_imputed_data]:
        for col in df.columns:
            for key, (minv, maxv) in column_bounds.items():
                if key in col:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if maxv is not None:
                            df[col] = np.clip(df[col], minv, maxv)
                        else:
                            df[col] = np.maximum(df[col], minv)
    
    # Map JADAS and save for train
    train_jadas = data_preprocessing.mapping_jadas(train_imputed_data)
    train_jadas.to_csv(
        f'{output_dir_imp}/{folder}/missforest_dflt_train_{param}.csv',
        index=True
    )

    # Map JADAS and save for test
    test_jadas = data_preprocessing.mapping_jadas(test_imputed_data)
    test_jadas.to_csv(
        f'{output_dir_imp}/{folder}/missforest_dflt_test_{param}.csv',
        index=True
    )
    
    
def my_missforest_cstm(train_data, test_data, param, output_dir_imp, folder='missforest'): 
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])
    
    categorical = ['male', 'jia_cat', 'insurance', 'income_cat', 'educate_cat',
                'race_amerind', 'race_asian', 'race_black', 'race_mideast', 'race_nathwn', 'race_white', 'hispanic',
                'hlab27', 'anticcp', 'famhx_rheum', 'famhx_ibd', 'famhx_uveitis', 'famhx_other_aid',
                'ibd', 'asthma', 'other_aid', 'iei']
    categorical+= ['totjoint_month.0', 'totjoint_month.6',
                'rad_damage_month.0',
                'uveitis_new_month.0', 'uveitis_new_month.6',
                'active_sacroilitis_month.0', 'active_sacroilitis_month.6',
                'active_enthesitis_month.0', 'active_enthesitis_month.6',
                'psoriasis_month.0', 'psoriasis_month.6',
                'newmed_month.0', 'newmed_month.6']
    categorical = [col for col in categorical if col in train_data.columns]

    column_bounds = {
        'PHYSCORE_month': (0, 10),
        'RHEURATE_month': (0, 10),
        'ACTJONUM2_month': (0, 10),
        'GHA_T_SCORE_month': (20, 80),
        'DIS_ACT_month': (0, 10),
        'RHEUM_PAIN_month': (0, 10)
        # 'diff_dx_symp_mth': (0, None)
    }
                
    mf = MissForest(
        clf=RandomForestClassifier(n_jobs=-1), 
        rgr=RandomForestRegressor(n_jobs=-1), 
        early_stopping=False,
        categorical = categorical
    )

    train_imputed_data = mf.fit_transform(train_data)
    test_imputed_data= mf.transform(test_data)

    for df in [train_imputed_data, test_imputed_data]:
        for col in df.columns:
            for key, (minv, maxv) in column_bounds.items():
                if key in col:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if maxv is not None:
                            df[col] = np.clip(df[col], minv, maxv)
                        else:
                            df[col] = np.maximum(df[col], minv)
    
    # Map JADAS and save for train
    train_jadas = data_preprocessing.mapping_jadas(train_imputed_data)
    train_jadas.to_csv(
        f'{output_dir_imp}/{folder}/missforest_cstm_train_{param}.csv',
        index=True
    )

    # Map JADAS and save for test
    test_jadas = data_preprocessing.mapping_jadas(test_imputed_data)
    test_jadas.to_csv(
        f'{output_dir_imp}/{folder}/missforest_cstm_test_{param}.csv',
        index=True
    )

#   DOESN'T WORK FOR CAT AND CONT VARS AT THE SAME TIME, essectially become the same as missforest if cat vars
#   FANCYIMPUTER PACKAGE IS WORTH EXPLORING
def my_iterative_imputer(train_data, test_data, param, output_dir_imp, folder): 
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])
    
    estimators = {
        "BayesianRidge": BayesianRidge(),
        "RandomForest": RandomForestRegressor(
            n_jobs=-1,
            random_state=0
        )
    }

    for name, est in estimators.items():
        bounds_dict = {
            'PHYSCORE': (0, 10),
            'RHEURATE': (0, 10),
            'ACTJONUM2': (0, 10),
            'educate_cat': (1, 5),
            'income_cat': (1, 6),
            'GHA_T_SCORE': (20, 80),
            'DIS_ACT': (0, 10),
            'RHEUM_PAIN': (0, 10)
        }
        
        min_values = []
        max_values = []

        for col in train_data.columns:
            matched=False
            for prefix, (min_val, max_val) in bounds_dict.items():
                if col.startswith(prefix):
                    min_values.append(min_val)
                    max_values.append(max_val)
                    matched = True
                    break
            if not matched:
                min_values.append(-np.inf)
                max_values.append(np.inf)
            
        imputer = IterativeImputer(estimator=est,
                                   min_value=min_values,
                                   max_value=max_values, 
                                   random_state=0, 
                                   max_iter=25)
        train_imputed = imputer.fit_transform(train_data)
        train_imputed = pd.DataFrame(train_imputed, columns=train_data.columns)
        
        data_imputed_jadas = data_preprocessing.mapping_jadas(train_imputed)  
        data_imputed_jadas.to_csv(
            f'{output_dir_imp}/{folder}/iterimputer_{name}_train_{param}.csv',
            index=True)
        
        
        
        test_imputed = imputer.transform(test_data)
        test_imputed = pd.DataFrame(test_imputed, columns=test_data.columns, index=test_data.index)

        # Map JADAS for test
        test_imputed_jadas = data_preprocessing.mapping_jadas(test_imputed)
        test_imputed_jadas.to_csv(
            f'{output_dir_imp}/{folder}/iterimputer_{name}_test_{param}.csv',
            index=True)

# DOESN'T WORK FOR CATEGORICAL VARIABLES!
def em_method(train_data, test_data, param, output_dir_imp, folder):
    """
    EM method implementation with log-transform -> impute -> exp-transform.
    """
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')
    numeric_cols = train_data.columns  

    # Log transform
    log_train = np.log(train_data[numeric_cols])
    log_test = np.log(test_data[numeric_cols])

    np.float = float
    log_train = log_train.values.astype(float)
    log_test = log_test.values.astype(float)

    log_train_imp = impyute.imputation.cs.em(log_train, loops=50) #.values
    log_test_imp = impyute.imputation.cs.em(log_test, loops=50) #.values

    # Exponentiate back to original scale
    train_imp_em = np.exp(log_train_imp)
    test_imp_em = np.exp(log_test_imp)

    train_imp_em_df = pd.DataFrame(train_imp_em, columns=numeric_cols, index=train_data.index)
    test_imp_em_df = pd.DataFrame(test_imp_em, columns=numeric_cols, index=test_data.index)

    train_imp_em_jadas = data_preprocessing.mapping_jadas(train_imp_em_df)
    train_imp_em_jadas.to_csv(
        f'{output_dir_imp}/{folder}/em_method_train_{param}.csv',
        index=True)

    test_imp_em_jadas = data_preprocessing.mapping_jadas(test_imp_em_df)
    test_imp_em_jadas.to_csv(
        f'{output_dir_imp}/{folder}/em_method_test_{param}.csv',
        index=True
    )

    
def my_midaspy(train_data, test_data, param, output_dir_imp, folder):
    """
    MIDASpy method implementation.
    """
    column_bounds = {
        'PHYSCORE_month': (0, 10),
        'RHEURATE_month': (0, 10),
        'ACTJONUM2_month': (0, 10),
        'GHA_T_SCORE_month': (20, 80),
        'DIS_ACT_month': (0, 10),
        'RHEUM_PAIN_month': (0, 10),
        'antiIL1_month': (0, 30),
        'antiIL6_month': (0, 30),
        'TNFi_month': (0, 30),
        'methotrexate_month': (0, 30),
        'otherBiologics_month': (0, 30),
        'otherDMARDs_mont': (0, 30),
        'smallMoleculeInhibitors_month': (0, 30)
    }

    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])
    
    cont_vars = [c for c in train_data.columns if 
                'age' in c or 'diff_dx_symp_mth' in c or 'jadas' in c or 'PHYSCORE' in c or
                 'RHEURATE' in c or 'ACTJONUM2' in c or 'antiIL1_month' in c or
                 'antiIL6_month' in c or 'methotrexate_month' in c or
                 'otherBiologics_month' in c or 'otherDMARDs_month' in c or
                 'smallMoleculeInhibitors_month' in c or
                 'TNFi_month' in c or 'GHA_T_SCORE_month' in c or 'DIS_ACT_month' in c or 
                 'RHEUM_PAIN_month' in c]

    bin_vars = [col for col in train_data.columns if 'anticcp' in col]

    cat_vars = [col for col in train_data.columns if 'totjoint_month' in col or 'rad_damage_month' in col or
                'uveitis_new_month' in col or 'active_sacroilitis_month' in col or 'active_enthesitis_month' in col or
                'psoriasis_month' in col or 'newmed_month' in col]
    cat_vars += ['jia_cat', 'insurance', 'income_cat', 'educate_cat', 'hlab27']

    """
    From the paper:
    First, binary variables must be converted to dummy (i.e., 0/1) form 
    without altering their missingness pattern
    Second, categorical variables must be one-hot encoded (i.e., divided into separate dummy
    variables for each unique class) while again preserving their missingness pattern.
    """
    
    train_bin = train_data[bin_vars].apply(md.binary_conv)
    test_bin  = test_data[bin_vars].apply(md.binary_conv)  

    single_val_col_bin = [col for col in train_data.columns if 'active_sacroilitis_month' in col]
    for col in single_val_col_bin:
        train_data[col] = train_data[col].replace({'Yes': 1, 'No': 0, 'Unknown': np.nan})
        test_data[col] = test_data[col].replace({'Yes': 1, 'No': 0, 'Unknown': np.nan})

    train_cat = train_data[cat_vars]
    train_oh, cat_col_list = md.cat_conv(train_cat)
    train_numerical = train_data.drop(columns=cat_vars+bin_vars)
    test_cat = test_data[cat_vars]
    test_oh, _ = md.cat_conv(test_cat)  
    test_numerical = test_data.drop(columns=cat_vars+bin_vars)
    
    train_0 = pd.concat([train_numerical, train_bin, train_oh], axis=1)
    test_0  = pd.concat([test_numerical, test_bin, test_oh], axis=1)


    # --- Check and align columns ---
    missing_in_test = set(train_0.columns) - set(test_0.columns)
    missing_in_train = set(test_0.columns) - set(train_0.columns)
    for col in missing_in_test:
        test_0[col] = np.nan
    for col in missing_in_train:
        train_0[col] = np.nan
    train_0 = train_0[sorted(train_0.columns)]
    test_0  = test_0[sorted(train_0.columns)]
    # -----------------------------------------------------

    train_id = train_0.index.copy()
    test_id  = test_0.index.copy()  # will be used to restore row index after imputation
    all_missing = test_0.isnull().all(axis=1)
    if all_missing.any():
        print(f"Warning: {all_missing.sum()} rows in test are all-NaN and will remain NaN after imputation!")

    train_0_reset = train_0.reset_index(drop=True)
    test_0_reset = test_0.reset_index(drop=True)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_0_reset)
    test_scaled = scaler.transform(test_0_reset)
    train_scaled = pd.DataFrame(train_scaled, columns=train_0.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=test_0.columns)
    train_scaled[train_scaled.isnull()] = np.nan
    test_scaled[test_scaled.isnull()]   = np.nan
    
    #layer_structure dflt [256,256,256]
    #input_drop = 0.8  is dflt, was 0.75
    imputer = md.Midas(vae_layer = False, seed= 89, savepath=output_dir_imp) 
    imputer.build_model(train_scaled, softmax_columns = cat_col_list)
    imputer.train_model() #training_epochs = 100 is dflt
    imputed_train = imputer.generate_samples(m=4).output_list

    imputer.change_imputation_target(test_scaled)
    imputed_test = imputer.generate_samples(m=4).output_list

    for i in range(4):
        imputed_train_unscaled = pd.DataFrame(
            scaler.inverse_transform(imputed_train[i].values), 
            columns=train_0.columns,
            index=train_id
        )
       
        imputed_test_unscaled = pd.DataFrame(
            scaler.inverse_transform(imputed_test[i].values), 
            columns=test_0.columns,
            index=test_id
        )

        flat_cats = [cat for variable in cat_col_list for cat in variable] 
        tmp_cat = [imputed_train_unscaled[x].idxmax(axis=1) for x in cat_col_list]
        cat_df = pd.DataFrame({cat_vars[j]:tmp_cat[j] for j in range(len(cat_vars))}, index=imputed_train_unscaled.index)
        imputed_train_final = pd.concat([imputed_train_unscaled, cat_df], axis = 1).drop(flat_cats, axis = 1)
        
        tmp_cat = [imputed_test_unscaled[x].idxmax(axis=1) for x in cat_col_list]
        cat_df = pd.DataFrame({cat_vars[j]:tmp_cat[j] for j in range(len(cat_vars))}, index=imputed_test_unscaled.index)
        imputed_test_final = pd.concat([imputed_test_unscaled, cat_df], axis = 1).drop(flat_cats, axis = 1)
        
        miss_train = imputed_train_final.isnull().sum().sum()
        miss_test = imputed_test_final.isnull().sum().sum()
        if miss_train > 0:
            print(f"[MIDASpy] Warning: {miss_train} missing values remain in train after imputation.")
        if miss_test > 0:
            nan_rows = imputed_test_final.isnull().all(axis=1).sum()
            print(f"[MIDASpy] Warning: {miss_test} missing values remain in test after imputation. {nan_rows} rows are all-NaN (cannot impute).")

        for df in [imputed_train_final, imputed_test_final]:
            for col in df.columns:
                for key, (minv, maxv) in column_bounds.items():
                    if key in col:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if maxv is not None:
                                df[col] = np.clip(df[col], minv, maxv)
                            else:
                                df[col] = np.maximum(df[col], minv)

        # JADAS mapping
        train_jadas = data_preprocessing.mapping_jadas(imputed_train_final)
        test_jadas  = data_preprocessing.mapping_jadas(imputed_test_final)

        train_jadas.to_csv(
            f'{output_dir_imp}/{folder}/midaspy_method_train_{param}_impute_{i+1}.csv',
            index=True
        )
        test_jadas.to_csv(
            f'{output_dir_imp}/{folder}/midaspy_method_test_{param}_impute_{i+1}.csv',
            index=True
        )

def apply_column_bounds(df, column_bounds):
    for col in df.columns:
        matching_bounds = [(key, bounds) for key, bounds in column_bounds.items() if key in col]
        for key, (minv, maxv) in matching_bounds:
            if pd.api.types.is_numeric_dtype(df[col]):
                if maxv is not None:
                    df[col] = np.clip(df[col], minv, maxv)
                else:
                    df[col] = np.maximum(df[col], minv)
    return df

def my_mice(train_data, test_data, param, output_dir_imp, folder):
    """
    MICE implementation.
    """
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])

    bin_vars = [col for col in train_data.columns if 'income_cat' in col or 'educate_cat' in col or
        'male' in col or 'race' in col or 'hispanic' in col or 'trisomy21' in col or 'famhx' in col or 
        'ibd' in col or 'asthma' in col or 'other_aid' in col or 'iei' in col or 'hlab27' in col or 
        'anticcp' in col or 'rad_damage_month' in col or 'uveitis_new' in col or 'active_enthesitis_month' in col or 
        'active_sacroilitis_month' in col or 'psoriasis_month' in col or 'newmed_month' in col]
    cat_vars = [col for col in train_data.columns if 'totjoint' in col]
    cat_vars += ['jia_cat', 'insurance', 'income_cat', 'educate_cat']

    single_val_col_bin = [col for col in train_data.columns if 'active_sacroilitis_month' in col]
    for col in single_val_col_bin:
        train_data[col] = train_data[col].replace({'Yes': 1, 'No': 0, 'Unknown': np.nan})
        test_data[col] = test_data[col].replace({'Yes': 1, 'No': 0, 'Unknown': np.nan})

    bin_cat = bin_vars + cat_vars

    for col in bin_cat:
        train_data[col] = train_data[col].astype("category")
        test_data[col] = test_data[col].astype("category")

    cols_to_drop = [col for col in train_data.columns if train_data[col].dropna().nunique() <= 1]
    # Store the value for each degenerate column
    degenerate_values = {}
    for col in cols_to_drop:
        vals = train_data[col].dropna().unique()
        if len(vals) > 0:
            degenerate_values[col] = vals[0]
        else:
            degenerate_values[col] = np.nan

    if cols_to_drop:
        print(f"Dropped degenerate columns in fold: {cols_to_drop}")
        train_data = train_data.drop(columns=cols_to_drop)
        test_data = test_data.drop(columns=cols_to_drop)
    original_columns = list(train_data.columns) + cols_to_drop

    column_bounds = {
        'PHYSCORE': (0, 10),
        'RHEURATE': (0, 10),
        'ACTJONUM2': (0, 10),
        'GHA_T_SCORE': (20, 80),
        'DIS_ACT': (0, 10),
        'RHEUM_PAIN': (0, 10),
        # 'diff_dx_symp_mth': (0, None)
    }

    mean_match_strategy = {col: ("fast" if col in bin_cat else "normal") for col in train_data.columns}
    kds = mf.ImputationKernel(
        train_data,
        num_datasets=4,
        save_all_iterations_data=True,
        random_state=0,
        mean_match_strategy=mean_match_strategy
    )

    # Ensures column order identical
    test_data = test_data[train_data.columns]

    # For each categorical column, force dtype and categories to match
    for col in train_data.select_dtypes(include="category").columns:
        test_data[col] = test_data[col].astype("category")
        test_data[col] = test_data[col].cat.set_categories(train_data[col].cat.categories)

    kds.mice(iterations=25) #, variable_parameters=optimal_params
    for i in range(4):
        imputed_train_final = kds.complete_data(dataset=i, inplace=False) 
        imputed_train_final = apply_column_bounds(imputed_train_final, column_bounds)

        # --- Add back degenerate columns ---
        for col, value in degenerate_values.items():
            imputed_train_final[col] = value
        imputed_train_final = imputed_train_final.reindex(columns=original_columns)
        # -------------------------------------

        train_jadas = data_preprocessing.mapping_jadas(imputed_train_final)
        train_jadas.to_csv(
        f'{output_dir_imp}/{folder}/mice_method_train_{param}_{i}.csv', index=True)
        del imputed_train_final
        del train_jadas
        gc.collect()
        

        imputed_test_final = kds.impute_new_data(new_data=test_data, random_state=0)
        imputed_test_final_df = imputed_test_final.complete_data()
        imputed_test_final_df = apply_column_bounds(imputed_test_final_df, column_bounds)

        # --- Add back degenerate columns ---
        for col, value in degenerate_values.items():
            imputed_test_final_df[col] = value
        imputed_test_final_df = imputed_test_final_df.reindex(columns=original_columns)
        # -------------------------------------

        test_jadas  = data_preprocessing.mapping_jadas(imputed_test_final_df)
        test_jadas.to_csv(
            f'{output_dir_imp}/{folder}/mice_method_test_{param}_{i}.csv',
            index=True
        )
        del imputed_test_final
        del imputed_test_final_df
        del test_jadas
        gc.collect()