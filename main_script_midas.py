import my_imp_methods_midas
import tensorflow as tf
import pandas as pd
from IPython.display import display
import numpy as np
import data_preprocessing
import functions_to_handle_data_Eleanor
from sklearn.ensemble import RandomForestClassifier

def extract_types(organized_data):
    organized_data_subset = []
    for fold_dict in organized_data:
        # Each fold_dict: {"train_i": train_dict, "test_i": test_dict}
        new_fold_dict = {}
        for set_name, set_dict in fold_dict.items():  # set_name: "train_1", etc. set_dict: {type_1:..., type_2:..., type_3:..., type_4:...}
            new_set_dict = {}
            # type_2: take only first (index 0) and second (index 1) elements, if available
            if "type_2" in set_dict:
                new_set_dict["type_2"] = set_dict["type_2"][:2]
            # type_3: take only first (index 0) element (corresponds to (0,6)), if available
            if "type_3" in set_dict:
                new_set_dict["type_3"] = set_dict["type_3"][:1]
            new_fold_dict[set_name] = new_set_dict
        organized_data_subset.append(new_fold_dict)
    return organized_data_subset

def main():
    _basepath = '/home/ch268692/pjia_proj/models_full_dataset'
    _filepath = '/home/ch268692/pjia_proj/data'
    _filename = 'data_eleanor_5Aug2025.csv'
    _filename_mapped = 'data_mapped_cols_not_enc.csv'
    _filename_mapped_enc = 'data_mapped_cols_enc.csv'
    output_dir_imp = '/home/ch268692/pjia_proj/models_full_dataset/imp_data'
    
    data_hyperparam = {'basepath': _basepath, 'filepath': _filepath, 'filename': _filename, 'filename_mapped': _filename_mapped, 
                    'med_boolean':True, 'encoding_flag': False}
    data_hyperparam_enc = {'basepath': _basepath, 'filepath': _filepath, 'filename': _filename, 'filename_mapped': _filename_mapped_enc, 
                    'med_boolean':True, 'encoding_flag': True}
    organized_data = functions_to_handle_data_Eleanor.create_kfold_splits(data_hyperparam, n_splits=5, output_dir="../fold_subject_ids")
    organized_data_enc = functions_to_handle_data_Eleanor.create_kfold_splits(data_hyperparam_enc, n_splits=5, output_dir="../fold_subject_ids")
    organized_data_subset = extract_types(organized_data)
    organized_data_enc_subset = extract_types(organized_data_enc)

    # Dictionary of all the methods
    imputation_methods = {
        'miss_forest_dflt': my_imp_methods_midas.my_missforest_dflt,
        'miss_forest_cstm': my_imp_methods_midas.my_missforest_cstm,
        'my_midaspy': my_imp_methods_midas.my_midaspy,
        'my_mice': my_imp_methods_midas.my_mice
        }
    
    for method_key, method in imputation_methods.items():
        if method_key != 'my_midaspy' and method_key != 'my_mice':
            curr_organized_data = organized_data_enc_subset
        else:
            curr_organized_data = organized_data_subset

        for fold_idx, fold_dict in enumerate(curr_organized_data):
            print(f'Fold:{fold_idx}')
            # --- type_2 ---
            train_data_type2 = fold_dict['train']['type_2']  # List: two elements (0, 6)
            test_data_type2 = fold_dict['test']['type_2']
            for i in range(len(train_data_type2)):  # Should be two: 0 and 6
                    
                print(f'Running {method_key} for fold {fold_idx+1}, type_2, dataset {i*6} (train/test)...')
                method(train_data_type2[i].copy(), test_data_type2[i].copy(), f'fold_{fold_idx+1}_type2_{i*6}', output_dir_imp, method_key)
            # --- type_3 ---
            train_data_type3 = fold_dict['train']['type_3']  # List: one element (0_6)
            test_data_type3 = fold_dict['test']['type_3']

            print(f'Running {method_key} for fold {fold_idx+1}, type_3, dataset 0_6 (train/test)...')
            method(train_data_type3[0].copy(), test_data_type3[0].copy(), f'fold_{fold_idx+1}_type3', output_dir_imp, method_key)
        
    print('Yay method ran successfully!')

if __name__ == "__main__":
    main()