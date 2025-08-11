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
from datawig import Imputer, CategoricalEncoder, EmbeddingFeaturizer, NumericalEncoder, NumericalFeaturizer, SequentialEncoder, LSTMFeaturizer

def safe_set_categorical(df, index, col, new_values):
    if pd.api.types.is_categorical_dtype(df[col]):
        imputed_vals_clean = np.array([np.nan if x == "nan" else x for x in new_values])
        existing_cats = set([cat for cat in df[col].cat.categories if pd.notnull(cat)])
        imputed_newvals = set([v for v in imputed_vals_clean if pd.notnull(v)])
        missing_cats = imputed_newvals - existing_cats
        if missing_cats:
            df[col] = df[col].cat.add_categories(list(missing_cats))
    df.loc[index, col] = new_values

def precompute_encoders_featurizers_labelencoders(df):
    data_encoders = {}
    data_featurizers = {}
    label_encoders = {}

    def is_text_column(s):
        s_non_null = s.dropna()
        # All values are str, at least some are long
        return (
            not s_non_null.empty and
            all(isinstance(x, str) for x in s_non_null) and
            s_non_null.str.len().max() > 30
        )

    for col in df.columns:
        col_dtype = df[col].dtype

        # Data encoder/featurizer
        if pd.api.types.is_categorical_dtype(col_dtype):
            data_encoders[col] = CategoricalEncoder(col)
            data_featurizers[col] = EmbeddingFeaturizer(col)
            label_encoders[col] = CategoricalEncoder(col)
        elif pd.api.types.is_numeric_dtype(col_dtype):
            data_encoders[col] = NumericalEncoder(col)
            data_featurizers[col] = NumericalFeaturizer(col)
            label_encoders[col] = NumericalEncoder(col)
        elif pd.api.types.is_object_dtype(col_dtype):
            # Text detection
            if is_text_column(df[col]):
                data_encoders[col] = SequentialEncoder(col)
                data_featurizers[col] = LSTMFeaturizer(col)
                label_encoders[col] = CategoricalEncoder(col) 
            else:
                data_encoders[col] = CategoricalEncoder(col)
                data_featurizers[col] = EmbeddingFeaturizer(col)
                label_encoders[col] = CategoricalEncoder(col)
        else:
            raise ValueError(f"Column {col} is of unsupported dtype: {col_dtype}")

    return data_encoders, data_featurizers, label_encoders



def build_datawig_imputer_from_precomputed(input_cols, output_col, output_path, data_encoders, data_featurizers, label_encoders):
    imputer = Imputer(
        data_encoders=[data_encoders[c] for c in input_cols],
        data_featurizers=[data_featurizers[c] for c in input_cols],
        label_encoders=[label_encoders[output_col]],
        output_path=output_path
    )
    return imputer


def my_datawig(train_data, test_data, param, output_dir_imp, output_path_prefix='imputer_model', folder = 'datawig'):
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])    

    categorical = ['male', 'jia_cat', 'insurance', 'income_cat', 'educate_cat',
                'race_amerind', 'race_asian', 'race_black', 'race_mideast', 'race_nathwn', 'race_white', 'hispanic',
                'hlab27', 'anticcp', 'famhx_rheum', 'famhx_ibd', 'famhx_uveitis', 'famhx_other_aid',
                'ibd', 'asthma', 'other_aid', 'iei', 'trisomy21']
    categorical += ['totjoint_month.0', 'totjoint_month.6',
                'rad_damage_month.0', 
                'uveitis_new_month.0', 'uveitis_new_month.6',
                'active_sacroilitis_month.0', 'active_sacroilitis_month.6',
                'active_enthesitis_month.0', 'active_enthesitis_month.6',
                'psoriasis_month.0', 'psoriasis_month.6',
                'newmed_month.0', 'newmed_month.6']
    categorical = [col for col in categorical if col in train_data.columns]


    for col in categorical:
        for d in [train_data, test_data]:
            if col in d.columns:
                d[col] = d[col].astype(str)  
                d[col] = d[col].replace('nan', np.nan) 
                d[col] = d[col].astype('category')
                if "" not in d[col].cat.categories: # Datawig feature
                    d[col] = d[col].cat.add_categories([""])

    train_imputed = train_data.copy()
    test_imputed = test_data.copy()
    train_working = train_data.copy() 
    test_working = test_data.copy()

    data_encoders, data_featurizers, label_encoders = precompute_encoders_featurizers_labelencoders(train_data)

    column_bounds = {
        'PHYSCORE': (0, 10),
        'RHEURATE': (0, 10),
        'ACTJONUM': (0, 10),
        'GHA_T_SCORE': (20, 80),
        'DIS_ACT': (0, 10),
        'RHEUM_PAIN': (0, 10),
        # 'diff_dx_symp_mth': (0, None)
    }
    
    for col in train_data.columns:
        if train_data[col].isnull().all():
            print(f"All values in {col} are NaN")

        if train_data[col].isnull().any() or test_data[col].isnull().any():
            print(f"Imputing column: {col}")
            input_cols = [c for c in train_data.columns if c != col]
            
            train_col_notnull = train_data[train_data[col].notnull()]
            train_col_notnull_reset = train_col_notnull.reset_index()
            if train_col_notnull_reset.empty:
                print(f"Skipping column '{col}' (empty DataFrame after reset_index)")
                continue

            imputer = build_datawig_imputer_from_precomputed(
                input_cols,
                col,
                output_path_prefix,
                data_encoders,
                data_featurizers,
                label_encoders
            )

            print(f"Training DataWig model for '{col}' with {len(train_col_notnull_reset)} rows...")

            imputer.fit(train_df=train_col_notnull_reset)  #, num_epochs=10
            train_col_null = train_data[train_data[col].isnull()]
            if not train_col_null.empty:
                pred_train = imputer.predict(train_col_null.reset_index())
                imputed_vals = pred_train[col + "_imputed"].values
                matching_bounds = [(key, bounds) for key, bounds in column_bounds.items() if key in col]
                for key, (minv, maxv) in matching_bounds:
                    imputed_vals = np.clip(imputed_vals, minv, maxv) if maxv is not None else np.maximum(imputed_vals, minv)
                # Update imputed values in both imputed and working df!
                safe_set_categorical(train_imputed, train_col_null.index, col, imputed_vals)
                safe_set_categorical(train_working, train_col_null.index, col, imputed_vals)

            # Predict missing in TEST set (using model fitted on train)
            test_col_null = test_data[test_data[col].isnull()]
            if not test_col_null.empty: 
                pred_test = imputer.predict(test_col_null.reset_index())
                imputed_vals_test = pred_test[col + "_imputed"].values
                matching_bounds = [(key, bounds) for key, bounds in column_bounds.items() if key in col]
                for key, (minv, maxv) in matching_bounds:
                    imputed_vals_test = np.clip(imputed_vals_test, minv, maxv) if maxv is not None else np.maximum(imputed_vals_test, minv)

                safe_set_categorical(test_imputed, test_col_null.index, col, imputed_vals_test)
                safe_set_categorical(test_working, test_col_null.index, col, imputed_vals_test)
            
    train_jadas = data_preprocessing.mapping_jadas(train_imputed)
    train_jadas.to_csv(
        f'{output_dir_imp}/{folder}/datawig_method_train_{param}.csv',
        index=True)
        
    test_jadas = data_preprocessing.mapping_jadas(test_imputed)
    test_jadas.to_csv(
        f'{output_dir_imp}/{folder}/datawig_method_test_{param}.csv',
        index=True)


def my_datawig_iterat(train_data, test_data, param, output_dir_imp, output_path_prefix='imputer_model_iterat', folder = 'datawig_iterat'):
    os.makedirs(f'{output_dir_imp}/{folder}', exist_ok=True)
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    cols_to_drop = ['steroid_month.0', 'rad_damage_month.6']
    train_data = train_data.drop(columns=[col for col in cols_to_drop if col in train_data.columns])
    test_data = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])    

    categorical = ['male', 'jia_cat', 'insurance', 'income_cat', 'educate_cat',
                'race_amerind', 'race_asian', 'race_black', 'race_mideast', 'race_nathwn', 'race_white', 'hispanic',
                'hlab27', 'anticcp', 'famhx_rheum', 'famhx_ibd', 'famhx_uveitis', 'famhx_other_aid',
                'ibd', 'asthma', 'other_aid', 'iei', 'trisomy21']
    categorical += ['totjoint_month.0', 'totjoint_month.6',
                'rad_damage_month.0', 
                'uveitis_new_month.0', 'uveitis_new_month.6',
                'active_sacroilitis_month.0', 'active_sacroilitis_month.6',
                'active_enthesitis_month.0', 'active_enthesitis_month.6',
                'psoriasis_month.0', 'psoriasis_month.6',
                'newmed_month.0', 'newmed_month.6']
    categorical = [col for col in categorical if col in train_data.columns]


    for col in categorical:
        for d in [train_data, test_data]:
            if col in d.columns:
                d[col] = d[col].astype(str)  
                d[col] = d[col].replace('nan', np.nan) 
                d[col] = d[col].astype('category')
                if "" not in d[col].cat.categories: # Datawig feature
                    d[col] = d[col].cat.add_categories([""])

    train_imputed = train_data.copy()
    test_imputed = test_data.copy()
    train_working = train_data.copy() 
    test_working = test_data.copy()

    data_encoders, data_featurizers, label_encoders = precompute_encoders_featurizers_labelencoders(train_data)
    
    column_bounds = {
        'PHYSCORE': (0, 10),
        'RHEURATE': (0, 10),
        'ACTJONUM': (0, 10),
        'GHA_T_SCORE': (20, 80),
        'DIS_ACT': (0, 10),
        'RHEUM_PAIN': (0, 10),
        # 'diff_dx_symp_mth': (0, None)
    }

    for col in train_data.columns:
        if train_working[col].isnull().all():
            print(f"All values in {col} are NaN")
            continue
        if train_working[col].isnull().any() or test_working[col].isnull().any():
            print(f"Imputing column: {col}")
            input_cols = [c for c in train_working.columns if c != col]
            
            train_col_notnull = train_working[train_working[col].notnull()]
            train_col_notnull_reset = train_col_notnull.reset_index()
            if train_col_notnull_reset.empty:
                print(f"Skipping column '{col}' (empty DataFrame after reset_index)")
                continue

            imputer = build_datawig_imputer_from_precomputed(
                input_cols,
                col,
                output_path_prefix,
                data_encoders,
                data_featurizers,
                label_encoders
            )

            print(f"  Training DataWig_iterat model for '{col}' with {len(train_col_notnull_reset)} rows...")

            # fit on not-null target rows using up-to-date features!
            imputer.fit(train_df=train_col_notnull_reset)

            # Impute missing train values using latest working data
            train_col_null = train_working[train_working[col].isnull()]
            if not train_col_null.empty:
                pred_train = imputer.predict(train_col_null.reset_index())
                imputed_vals = pred_train[col + "_imputed"].values
                matching_bounds = [(key, bounds) for key, bounds in column_bounds.items() if key in col]
                for key, (minv, maxv) in matching_bounds:
                    imputed_vals = np.clip(imputed_vals, minv, maxv) if maxv is not None else np.maximum(imputed_vals, minv)
                if pd.api.types.is_categorical_dtype(train_imputed[col]):
                    missing_cats = set(imputed_vals) - set(train_imputed[col].cat.categories)
                    if missing_cats:
                        train_imputed[col] = train_imputed[col].cat.add_categories(list(missing_cats))
                # Update imputed values in both imputed and working df
                train_imputed.loc[train_col_null.index, col] = imputed_vals
                train_working.loc[train_col_null.index, col] = imputed_vals  

            # Impute missing test values using up-to-date features
            test_col_null = test_working[test_working[col].isnull()]
            if not test_col_null.empty: 
                pred_test = imputer.predict(test_col_null.reset_index())
                imputed_vals_test = pred_test[col + "_imputed"].values
                matching_bounds = [(key, bounds) for key, bounds in column_bounds.items() if key in col]
                for key, (minv, maxv) in matching_bounds:
                    imputed_vals_test = np.clip(imputed_vals_test, minv, maxv) if maxv is not None else np.maximum(imputed_vals_test, minv)
                if pd.api.types.is_categorical_dtype(test_imputed[col]):
                    missing_cats = set(imputed_vals_test) - set(test_imputed[col].cat.categories)
                    if missing_cats:
                        test_imputed[col] = test_imputed[col].cat.add_categories(list(missing_cats))
                test_imputed.loc[test_col_null.index, col] = imputed_vals_test
                test_working.loc[test_col_null.index, col] = imputed_vals_test
            
    train_jadas = data_preprocessing.mapping_jadas(train_imputed)
    train_jadas.to_csv(
        f'{output_dir_imp}/{folder}/datawig_iterat_method_train_{param}.csv',
        index=True)
        
    test_jadas = data_preprocessing.mapping_jadas(test_imputed)
    test_jadas.to_csv(
        f'{output_dir_imp}/{folder}/datawig_iterat_method_test_{param}.csv',
        index=True)