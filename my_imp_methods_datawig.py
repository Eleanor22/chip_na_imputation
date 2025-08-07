import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import os
from sklearn.model_selection import train_test_split
import warnings
import data_preprocessing #import mapping_jadas
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler
import datawig



def my_datawig(train_data, test_data, param, output_dir_imp, 
output_path_prefix='imputer_model', folder = 'datawig'):
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])

    train_imputed = train_data.copy()
    test_imputed = test_data.copy()
    
    for col in train_data.columns:
        if train_data[col].isnull().any() or test_data[col].isnull().any():
            print(f"Imputing column: {col}")
        
        if train_data[col].isnull().all():
            print(f"All values in {col} are NaN")
            
            input_cols = [c for c in train_data.columns if c != col]
            
        
            train_col_notnull = train_data[train_data[col].notnull()]
            train_col_notnull_reset = train_col_notnull.reset_index()
            if train_col_notnull_reset.empty:
                print(f"Skipping column '{col}' (empty DataFrame after reset_index)")
                continue


            os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)

            imputer = datawig.SimpleImputer(
                input_columns=input_cols,
                output_column=col,
                output_path=f"{output_path_prefix}"
            )

            print(f"  Training DataWig model for '{col}' with {len(train_col_notnull_reset)} rows...")
            imputer.fit(train_df=train_col_notnull_reset, num_epochs=10)
            
            train_col_null = train_data[train_data[col].isnull()]
            if not train_col_null.empty:
                pred_train = imputer.predict(train_col_null.reset_index())
                train_imputed.loc[train_col_null.index, col] = pred_train[col + "_imputed"].values

            # Predict missing in TEST set (using model fitted on train)
            test_col_null = test_data[test_data[col].isnull()]
            if not test_col_null.empty: 
                pred_test = imputer.predict(test_col_null.reset_index())
                test_imputed.loc[test_col_null.index, col] = pred_test[col + "_imputed"].values
            
    train_jadas = data_preprocessing.mapping_jadas(train_imputed)
    train_jadas.to_csv(
        f'{output_dir_imp}/{folder}/datawig_method_train_{param}.csv',
        index=True)
        
    test_jadas = data_preprocessing.mapping_jadas(test_imputed)
    test_jadas.to_csv(
        f'{output_dir_imp}/{folder}/datawig_method_test_{param}.csv',
        index=True)