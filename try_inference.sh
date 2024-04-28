#!/bin/bash
#SBATCH --job-name=IFM
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --time=3:00:00
#SBATCH --output=/home/sh2748/Logs/IFM/protein/expt_%J.log

date;hostname;pwd
module load miniconda
conda activate fm

cd /home/sh2748/protein-frame-flow

python -W ignore experiments/inference_se3_flows.py