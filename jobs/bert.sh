#!/bin/bash
# JOB HEADERS HERE
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eurb@umich.edu,agedeon@umich.edu,anrao@umich.edu,wongna@umich.edu,bsteinig@umich.edu
#SBATCH --job-name=emojiComplete

# Starts in the jobs folder. Need to bring it back
cd ..

# Load correct python version
module load python/3.9.7

python src/bert.py train False