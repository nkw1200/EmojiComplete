#!/bin/bash
# JOB HEADERS HERE
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eurb@umich.edu,agedeon@umich.edu,anrao@umich.edu,wongna@umich.edu,bsteinig@umich.edu
#SBATCH --job-name=emojiComplete

python src/bert.py train False