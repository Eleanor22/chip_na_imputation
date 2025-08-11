from os import listdir
import re
from os import path as osp
import pandas as pd
import numpy as np
import data_preprocessing
from IPython.display import display
from sklearn.model_selection import KFold, StratifiedKFold
import os

# !!!!!!!!!--IMPORTANT NOTES--!!!!!!!!!!!
# rad_damage only has baseline, no 6 month data 
# psoriasis only has 6 month, no baseline


def data_converter_dynamics(filepath, filename):
    # Convert the name of the columns for easier manipulation
    file = osp.join(filepath, filename)
    if isinstance(file, str):
        if 'csv' in file[-3:]:
            df = pd.read_csv(file, low_memory=False)
        elif 'xlsx' in file[-4:]:
            df = pd.read_excel(file)
        else:
            raise NotImplementedError("\tCurrently, the supported datafile extensions are csv and xlsx!")
    scores_type = ['jadas', 'PHYSCORE', 'RHEURATE', 'ACTJONUM2']
    columns_1type = ['totjoint', 'rad_damage', 'uveitis_new', 'active_sacroilitis', 
                    'active_enthesitis', 'psoriasis', 'newmed'] #'si_arthritis', 'sacroiliac_tenderness', 'acute_uveitis', 'inflam_backpain', 'abnormal_esr',
                                                                #'abnormal_creaprot', 'si_damage'
    columns_2type = ['steroid']
    another_scores_type = ['GHA_T_SCORE', 'DIS_ACT', 'RHEUM_PAIN']
    
    time_map_1 = {
        'baseline': '0',
        '_1_6m': '6',
        }
    
    time_map_2 = {
        '_first_6m': '6',
        }
    
    time_map_3 = {
        '0': '0',
        '6m': '6', 
        }
    
    
    mapped_columns = {}
    
    for col in df.columns:
        # print(f'Column: {col}')
        matched=False
        for key in columns_1type:
            if key in col:
                for pattern, month_val in time_map_1.items():
                    if pattern in col:
                        new_colname = f"{key}_month.{month_val}"
                        mapped_columns[col] = new_colname
                        matched=True
        if matched:
            continue
            
    
            
        for key in columns_2type:
            if key in col:
                for pattern, month_val in time_map_2.items():
                    if pattern in col: #column
                        new_colname = f"{key}_month.{month_val}"
                        mapped_columns[col] = new_colname
                        matched=True
        if matched:
            continue
        
        for key in another_scores_type:
            for pattern, month_val in time_map_3.items():
                if col == f"{key}.{pattern}" or col == f"{key}_{pattern}":
                    mapped_columns[col] = f"{key}_month.{month_val}"
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
                        
        for key in scores_type:
            if col.startswith(f"{key}.") or f"{key}." in col:
                # Only match if there's a dot followed by a number
                if re.match(rf"^{key}\.\d+$", col):
                    new_colname = re.sub(r'\.(\d+)$', r'_month.\1', col)
                    mapped_columns[col] = new_colname
                    break                    
    
    df = df.rename(columns=mapped_columns)

    # print(f"Unique values in column 'active_sacroilitis_month.0': {df['active_sacroilitis_month.0'].unique()}")
    
    return df 
        
         

def data_loader_dynamics(data_hyperparam):
    _BASEPATH = data_hyperparam['basepath']
    _FILEPATH = data_hyperparam['filepath']
    datafilename = data_hyperparam['filename']
    datafilename_mapped = data_hyperparam['filename_mapped']
    use_demographic_var = True
    medication_bool = data_hyperparam['med_boolean']
    encoding_flag = data_hyperparam['encoding_flag']

    ## Load datafile
    if datafilename_mapped in listdir(_BASEPATH):
        # load previously saved converted datafile
        df = pd.read_csv(osp.join(_BASEPATH, datafilename_mapped), low_memory=False)
        print("Successfully loaded the pre-processed datafile!")
    elif datafilename in listdir(_FILEPATH):
        # load raw data and process the data
        if 'csv' in datafilename[-3:]:
            df = pd.read_csv(osp.join(_FILEPATH, datafilename), low_memory=False)
        elif 'xlsx' in datafilename[-4:]:
            df = pd.read_excel(osp.join(_FILEPATH, datafilename))
        else:
            raise NotImplementedError("\tCurrently, the supported datafile extensions are csv and xlsx!")
        df = data_converter_dynamics(_FILEPATH, datafilename)

        if encoding_flag:
            name = 'enc'
            df = data_preprocessing.data_preprocessing(df, flag=True)
        else:
            name = 'not_enc'
            df = data_preprocessing.data_preprocessing(df, flag=False)
        if 'steroid_month.0' not in df.columns:
            df['steroid_month.0'] = np.zeros(df.shape[0]) # No zero month column, so adding a dummy column
        if 'rad_damage_month.6' not in df.columns: 
            df['rad_damage_month.6'] = np.zeros(df.shape[0])  # No 6 month column, so adding a dummy column

        
        # ---------- Conversion block is HERE ----------
        cols_to_float_base = [
            'GHA_T_SCORE_month.',
            'DIS_ACT_month.',
            'RHEUM_PAIN_month.',
        ]
        cols_to_float = [f"{prefix}{6*tcnt}" for prefix in cols_to_float_base for tcnt in range(7)]
        for col in cols_to_float:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # ---------- END conversion block ----------
        

        df.to_csv(osp.join(_BASEPATH, f'data_mapped_cols_{name}.csv'), na_rep='NA', index=False)
        print("Successfully pre-processed the raw data!")
    else:
        raise ValueError("Given datafile doesn't exist in Data directory!")

    
    ## Collect x and y
    state_name_list = [] 
    for tcnt in range(7): 
        state_name_list = state_name_list + [ 
            'totjoint_month.%d'%(6*tcnt),
            'rad_damage_month.%d'%(6*tcnt),
            'uveitis_new_month.%d'%(6*tcnt),
            'active_sacroilitis_month.%d'%(6*tcnt),
            'active_enthesitis_month.%d'%(6*tcnt),
            'psoriasis_month.%d'%(6*tcnt),
            'newmed_month.%d'%(6*tcnt),
            'steroid_month.%d'%(6*tcnt), #No zero month column in the initial data
            'PHYSCORE_month.%d'%(6*tcnt),
            'RHEURATE_month.%d'%(6*tcnt),
            'ACTJONUM2_month.%d'%(6*tcnt),
            'jadas_month.%d'%(6*tcnt),
            'GHA_T_SCORE_month.%d'%(6*tcnt),
            'DIS_ACT_month.%d'%(6*tcnt),
            'RHEUM_PAIN_month.%d'%(6*tcnt),          
        ] 


    demog_name_list = [
        'id',
        'male',
        'age',
        'jia_cat',
        'insurance',
        'income_cat',
        'educate_cat',
        'race_amerind',
        'race_asian',
        'race_black',
        'race_mideast',
        'race_nathwn',
        'race_white',
        'hispanic',
        'hlab27',
        'anticcp',
        'diff_dx_symp_mth',
        'trisomy21',
        'famhx_rheum',
        'famhx_ibd',
        'famhx_uveitis',
        'famhx_other_aid',
        'ibd',
        'asthma',
        'other_aid',
        'iei'
    ]

    medication_name_list = []
    for tcnt in range(1, 37):
        medication_name_list = medication_name_list + [
            'antiIL1_month.%d'%(tcnt),
            'antiIL6_month.%d'%(tcnt),
            'methotrexate_month.%d'%(tcnt),
            'otherBiologics_month.%d'%(tcnt),
            'otherDMARDs_month.%d'%(tcnt),
            'smallMoleculeInhibitors_month.%d'%(tcnt),
            'TNFi_month.%d'%(tcnt)
        ]


    state_names_list = [col for col in state_name_list if col in df.columns]
    demog_names_list = [col for col in demog_name_list if col in df.columns]
    medication_names_list = [col for col in medication_name_list if col in df.columns]

    if 'id' not in state_names_list:
        state_names_list = ['id'] + state_names_list
    if 'id' not in medication_names_list:
        medication_names_list = ['id'] + medication_names_list
    
    states = df[state_names_list] #.astype(float)
    demogs = df[demog_names_list] #.astype(float)
    meds = df[medication_names_list] #.astype(float)

    return {'states': states, 'demogs': demogs, 'meds': meds}


def data_organizer_dynamics_system(loaded_dataset, use_demog=True, twindow=[1, 2, 3]): 
    ## Reorganize the loaded data into training/evaluation data for ML models
    ## Input
    # dataset has keys: 'states', 'demogs', 'meds'
    # 'boolmed3' means that medications are boolean over every two months so 3 medication variables are related to prediction
    # i.e. (0th, 1st), (2nd, 3rd), (4th, 5th) -> prediction of 6th month
    ## Output
    # learning set: training and validation for cross-validation
    # evaluation set: organized data for the series of temporal predictions (feed neural net's prediction in 6th month as input to generate 12th month prediction, and iterate the process until 36th month)
    state_step = 1
    meds_per_state = 6

    data_states = loaded_dataset['states']
    data_demogs = loaded_dataset['demogs']
    data_meds = loaded_dataset['meds']
    num_samples = [data_states.shape[0], data_demogs.shape[0], data_meds.shape[0]]
    tmp_flags = [num_samples[a]==num_samples[a+1] for a in range(len(num_samples)-1)]
    assert all(tmp_flags), "Loaded data have different number of patients in sub-categories!"
    del tmp_flags
    num_samples, num_states, num_meds = num_samples[0], 15, 7 # NEEDS TO BE MODIFIED EVERY TIME ACCORDINGLY!
     
    organized_data = {
        'type_1': [],  # all timepoints
        'type_2': [],  # individual timepoints
        'type_3': [],  # overlapping windows (0,6), (6,12), etc.
        'type_4': []   # rolling 3-step windows (0,6,12), (6,12,18), etc.
    } 
    
    
    # -------- TYPE 1 -------- #
    full_df = pd.concat([data_states, data_meds], axis=1)
    if use_demog:
        full_df = pd.concat([full_df, data_demogs], axis=1)
    organized_data['type_1'].append(full_df)


    # -------- TYPE 2 -------- #
    for t in range(7):       
        state_cols = [col for col in data_states.columns if f"month.{6*t}" in col]
        # print(f'State_cols: {state_cols}')
        states_df = data_states[state_cols]

        if t == 0:
            tmp_df = states_df.copy()
        else:            
            start_month = 6 * (t - 1) + 1
            end_month = 6 * t
            months = range(start_month, end_month + 1)
            
            meds_cols = [col for col in data_meds.columns if any(col.endswith(f"month.{m}") for m in months)]
            meds_df = data_meds[meds_cols]
            tmp_df = pd.concat([states_df, meds_df], axis=1)

        if use_demog:
            tmp_df = pd.concat([tmp_df, data_demogs], axis=1)

        organized_data['type_2'].append(tmp_df)

    # -------- TYPE 3 -------- #
    for t in range(6):
        month_1 = 6 * t
        month_2 = 6 * (t + 1)

        # Get state columns for both months
        state_cols = [col for col in data_states.columns if f"month.{month_1}" in col or f"month.{month_2}" in col]
        states_df = data_states[state_cols]

        # Get meds only for month_2 (not the full window)
        if t == 0:
            meds_months = list(range(0, 6 + 1))  # months 0 to 6 inclusive
        elif t==1:
            meds_months = list(range(0, 12 + 1)) 
        else:
            meds_months = list(range(6 * (t - 1)+1, 6 * (t + 1)+1))  # e.g., 6 to 18 for t=2
        med_cols = [col for col in data_meds.columns if any(col.endswith(f"month.{m}") for m in meds_months)]
        meds_df = data_meds[med_cols]


        tmp_df = pd.concat([states_df, meds_df], axis=1)
        if use_demog:
            tmp_df = pd.concat([tmp_df, data_demogs], axis=1)
        organized_data['type_3'].append(tmp_df)


    # -------- TYPE 4 -------- #
    for t in range(5):
        # print(f't is {t}')
        state_months = [6 * t, 6 * (t + 1), 6 * (t + 2)]
        state_cols = [col for col in data_states.columns if any(f"month.{m}" in col for m in state_months)]
        states_df = data_states[state_cols]

        # --- Get meds from appropriate months ---
        if t == 0:
            meds_months = list(range(0,13))
        # elif t == 1:
        #     meds_months = list(range(0, 18 + 1))  # months 0 to 18
        else:
            meds_months = list(range(6 * (t - 1) + 1, 6 * (t + 2) + 1))  # e.g., 7 to 24 for t=2

        med_cols = [col for col in data_meds.columns if any(col.endswith(f"month.{m}") for m in meds_months)]
        meds_df = data_meds[med_cols]

        tmp_df = pd.concat([states_df, meds_df], axis=1)
        
        if use_demog:
            tmp_df = pd.concat([tmp_df, data_demogs], axis=1)
        organized_data['type_4'].append(tmp_df)

    return organized_data


def create_kfold_splits(data_hyperparam, n_splits=5, output_dir="../fold_subject_ids", organizer_kwargs=None, rs=1):
    """
    Splits the dataset into k folds by patient and organizes each fold's data in the desired format.

    Returns:
        splits: List of dicts [{'train':..., 'test':...}, ...] with organized data
        Writes subject IDs for each train/test split into files.
    """
    splits = []
    if organizer_kwargs is None:
        organizer_kwargs = {}

    loaded_data = data_loader_dynamics(data_hyperparam)
    # Must contain id
    # We'll use states, but demogs or meds would also work as long as the index is patient
    assert 'id' in loaded_data['demogs'].columns, "Subject ID column is required!"
    all_df = loaded_data['demogs']
    unique_subjects = all_df['id'].unique()
    unique_subjects = sorted(unique_subjects)  # deterministic
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=29) #3,9 worked for sacroiliac
    splits_idx = list(kf.split(unique_subjects))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for fold_i, (train_idx, test_idx) in enumerate(kf.split(unique_subjects)):
    for fold_i in range(n_splits):
        train_file = f"{output_dir}/fold_{fold_i}_train_subjects.csv"
        test_file = f"{output_dir}/fold_{fold_i}_test_subjects.csv"

        if os.path.exists(train_file) and os.path.exists(test_file):
            train_ids = pd.read_csv(train_file, header=None)[0].tolist()
            test_ids = pd.read_csv(test_file, header=None)[0].tolist()
            print(f"Loaded subject IDs for fold {fold_i} from files.")
        else:
            train_idx, test_idx = splits_idx[fold_i]
            train_ids = [unique_subjects[i] for i in train_idx]
            test_ids = [unique_subjects[i] for i in test_idx]
            pd.Series(train_ids).to_csv(train_file, index=False, header=False)
            pd.Series(test_ids).to_csv(test_file, index=False, header=False)
            print(f"Wrote subject IDs for fold {fold_i} to files.")
       
        # Filter the full data by train/test subject IDs
        def filter_subjects(df):
            return df[df['id'].isin(train_ids)].reset_index(drop=True), \
                   df[df['id'].isin(test_ids)].reset_index(drop=True)

        loaded_train = {} 
        loaded_test = {}
        for key in loaded_data:
            train_df, test_df = filter_subjects(loaded_data[key])
            loaded_train[key] = train_df
            loaded_test[key] = test_df

        # Organize data for train/test
        organized_data_train = data_organizer_dynamics_system(loaded_train, **organizer_kwargs)
        organized_data_test = data_organizer_dynamics_system(loaded_test, **organizer_kwargs)

        splits.append({'train': organized_data_train, 'test': organized_data_test})

    return splits