#!/bin/bash
#SBATCH --partition=bch-compute
#SBATCH --account=bch
#SBATCH --time=11:59:59
#SBATCH --job-name=data_imp_job_v1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eleanor.belkin@childrens.harvard.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10   
#SBATCH --mem=16GB


# >>>>>> Activate conda environment <<<<<<
source ~/.bashrc
conda activate /home/ch268692/.conda/envs/venv_datawig

# >>>>>> Run your Python script <<<<<<
python3 main_script_datawig.py   # <-- replace with your script name