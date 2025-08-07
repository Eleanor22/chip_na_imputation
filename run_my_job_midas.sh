#!/bin/bash
#SBATCH --partition=bch-compute
#SBATCH --time=11:59:59
#SBATCH --job-name=data_imp_job_v1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eleanor.belkin@childrens.harvard.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16   
#SBATCH --mem=64GB


# >>>>>> Activate conda environment <<<<<<
source ~/.bashrc
conda activate /home/ch268692/.conda/envs/venv_miss

# >>>>>> Run your Python script <<<<<<
python3 main_script_midas.py   # <-- replace with your script name