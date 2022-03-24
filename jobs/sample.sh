#!/bin/bash
# JOB HEADERS HERE
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=[INSERT EMAIL]
#SBATCH --job-name=emojiComplete
pip3 install matplotlib --user
pip3 install numpy --user
pip3 install scipy --user
cd /home/eurb/hw4
python3 digit_classification.py > out.log
