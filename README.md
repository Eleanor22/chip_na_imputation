<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#project-layout">Project layout</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li>
        <a href="#installation-and-execution">Installation and execution</a>
            <ul>
                <li><a href="#setting-up-the-environment-for-all-the-methods-but-datawig">Setting Up the Environment for all the methods but DataWig</a></li>
                <li><a href="#setting-up-the-environment-for-datawig">Setting Up the Environment for DataWig</a></li>
            </ul>
        </li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#configuration-notes">Configuration notes</a></li>
        <li><a href="#data-structure">Data Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#additional-guidelines-and-comments">Additional guidelines and comments</a>
    </li>
    <li>
      <a href="#potential-improvements">Potential improvements</a>
    </li>
    


  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

### Overview

This documentation provides a comprehensive guide to the tools and methodologies developed for processing and imputing missing values within the CARRA (Childhood Arthritis and Rheumatology Research Alliance) Registry dataset.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Project layout

There were implemented 8 imputation methods. They are:
1. Missing forest with the default setting of regressor/classifier (missforest_dflt) using MissForest
2. Missing forest with a customized setting of regressor/classifier (missforest_cstm) using MissForest
3. Iterative imputer (II) using IterativeImputer from sklearn
4. Expectation-Maximization (EM) using impyute
5. Multiple Imputation with Denoising Autoencoders (MIDAS) 
6. DataWig
7. Iterative Datawig
8. Multiple Imputation by Chained Equations (MICE) using miceforest

**Note:**
- While some methods share similar underlying ideas, they may differ in subtle yet important aspects—whether they perform single or multiple imputations, the type of classifiers and regressors they use. These distinctions affect implementation and might affect the results.
- The II and EM methods do not support categorical variables. While these methods may be suitable for certain types of datasets, moving forward, our focus will shift to the other six imputation methods that are compatible with categorical data.

**The project layout and description of each file is below.**
```plaintext
models_full_dataset/
├── data_preprocessing.py
├── functions_to_handle_data_Eleanor.py
├── my_imp_methods_midas.py
├── my_imp_methods_datawig.py
├── main.ipynb
├── main_script_midas.py
├── main_script_datawig.py
├── run_my_job_midas.sh
└── run_my_job_datawig.sh
```
| File/Filename                        | Description                                                                |
|-------------------------------------- |----------------------------------------------------------------------------|
| `data_preprocessing.py`              | Processing of certain categorical columns and function for the JADAS value calculation     |
| `functions_to_handle_data_Eleanor.py` | Organizing data into the right 5-CV format to run models                   |
| `my_imp_methods_midas.py`            | Implementation of all the methods except DataWig                           |
| `my_imp_methods_datawig.py`          | Implementation of the DataWig method                                       |
| `main.ipynb`                        | Interactive Jupyter notebook for testing                                   |
| `main_script_midas.py`              | Script to run all models except DataWig                                    |
| `main_script_datawig.py`            | Script to run the DataWig model                                            |
| `run_my_job_midas.sh`               | Shell script for all methods except DataWig to run on e3                   |
| `run_my_job_datawig.sh`             | Shell script for DataWig to run on e3                                      |



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This section outlines the environment requirements for running the imputation methods used in this project. \
**Important note:** All methods except Datawig can be run within the same environment. However, to run the Datawig method, you must create a separate environment because its required package versions conflict with those needed for the other methods, especially MIDAS. Below, you will find setup instructions for these two different environments.

### Installation and execution
#### Setting Up the Environment for all the methods but DataWig

The steps below are written for a Conda environment, but you can modify them for other virtual environment tools if needed.

1. Navigate to Your Project Directory
```sh
cd /path/to/your/project/directory
```
2. Create and activate a conda virtual environment
```sh
conda create -n myenv python=3.10.18
conda activate myenv
```
3. Install required packages
```sh
conda install pip
%pip install MIDAS
%pip install MissForest
%pip install lightgbm
%pip install tqdm
%pip install seaborn
%pip install impyute
%pip install miceforest
```
4. Running the script

Refer to the [Usage section](#usage) for further details before running the script.

- Run directly in Python:
```sh
python3 main_script_midas.py
```
- Or schedule a job on e3:
```sh
sbatch run_my_job_midas.sh
```
**Note:** For some reason, sometimes the above script `run_my_job_midas.sh` doesn't work and you need to delete `#SBATCH --account=bch` line from the file to make it work...

#### Setting Up the Environment for DataWig

1. Navigate to Your Project Directory
```sh
cd /path/to/your/project/directory
```
2. Create and activate a conda virtual environment
```sh
conda create -n myenv python=3.7.16
conda activate myenv
```
3. Install required packages
```sh
conda install pip
%pip install datawig
```
**Note:** there might be some additional packages that would be required to install, I forgot to document when I was setting up the environment...

4. Running the script

Refer to the [Usage section](#usage) for further details before running the script.

- Run directly in Python:
```sh
python3 main_script_datawig.py
```
- Or schedule a job on e3:
```sh
sbatch run_my_job_datawig.sh
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage
### Configuration notes
Before running the methods, it is important to make sure your environment and file paths are correctly set up. \
**Modify the following variables/files as needed for your system:**
- Shell Scripts

    Update the following in both `run_my_job_midas.sh` and `run_my_job_datawig.sh`:

    - Your email address (for notifications)
    - The conda environment path (where your environment resides)
    ```sh
    #SBATCH --mail-user=your_email@domain.com
    source /path/to/your/conda.sh
    conda activate /path/to/your/conda/env
    ```

- Main Python Scripts

    The following variable assignments are present in both `main_script_midas.py` and `main_script_datawig.py`. 
    ```python
    _basepath = None          # Path to your code directory
    _filepath = None          # Path to your data directory
    _filename = None                 # Main input data file
    _filename_mapped = None      # Mapped, not encoded data
    _filename_mapped_enc = None   # Mapped and encoded data
    output_dir_imp = None     # Output dir for imputed data
    ```

- Additional Python Scripts

    The following variable assignments in `functions_to_handle_data_Eleanor.py` have to be modified each time data changes.
    ```python
    num_samples, num_states, num_meds = num_samples[0], 16, 7
    ```

### Data Structure

The data is organized into five train/test folds for cross-validation. The structure in Python can be illustrated as:
```python
organized_data = [
    { 'train_1': ..., 'test_1': ... },
    { 'train_2': ..., 'test_2': ... },
    ...
    { 'train_5': ..., 'test_5': ... }
]
```
Within each fold, train_i and test_i are dictionaries with the following possible types of datasets:
- type_1:

    [ (all_states, all_meds, demogs) ] \
    Tuples containing all states, all medications, and demographic information.
- type_2:

    [ (data at 0), (data at 6), ..., (data at 36) ] \
    Data snapshots taken at each time point.

- type_3:
    [ (data at 0,6), (6,12), ..., (30,36) ] \
    Data grouped in two-interval windows.

- type_4:

    [ (data at 0,6,12), (6,12,18), ..., (24,30,36) ] \
    Data grouped in three-interval windows.

**Note:** The full data structure for all types is implemented, but for this project, only type_2 and type_3 datasets are used. The additional structure allows flexibility for future analyses.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Additional guidelines and comments -->
## Additional guidelines and comments
Below are additional guidelines and helpful comments:
- Data Directory: \
The data should be placed in a separate folder "data". 
- Fold Subject IDs: \
A folder named "fold_subject_ids" will be automatically created.
This folder will contain files listing patient IDs for each fold, used during training and testing of the models.
The naming convention for these files is:
    - `fold_i_train_subjects.csv` for training dataset
    - `fold_i_test_subjects.csv` for test dataset
- Mapped columns and encoding of the data: \
Two additional files will be automatically created:
    - `data_mapped_cols_enc.csv` 
    - `data_mapped_cols_not_enc.csv` 

    MIDAS, MICE, and Datawig have built-in functionality to encode data as required by the model; therefore, it uses the unencoded dataset (`data_mapped_cols_not_enc.csv`) as input. All other models require categorical variables to be pre-encoded, so they use the encoded dataset (`data_mapped_cols_enc.csv`). \
    If these files already exist, they will be used to rerun the models.
- Imputed Data Output: 
The resulting imputed data will be created in the automatically generated folder 'imp_data'.
The naming convention for these files is: `{name_of_the_method}_method_{test_or_train_dataset}_fold_{i}_{type_of_the_data}.csv`
- Multiple imputed datasets:
To quantify uncertainty, several datasets have been imputed using the MIDAS and MICE methods. For downstream analyses, results can be averaged across these multiple imputations, providing a more robust estimate by accounting for imputation uncertainty.
- Iterative Datawig method
The Datawig imputation method is not inherently iterative. As an alternative, we implemented an Iterative Datawig approach, where previously imputed columns are sequentially used to inform the imputation of subsequent columns. This procedure may lead to improved accuracy by leveraging information from earlier imputations.


## Potential improvements
- Consider tuning the hyperparameters of the imputation methods to potentially improve their performance. All methods used in this project have parameters that can be adjusted to achieve better results. 
