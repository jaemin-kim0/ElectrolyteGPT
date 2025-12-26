#!/bin/bash
#SBATCH -t 36:00:00
#SBATCH --exclusive
#SBATCH --partition=xxx
#SBATCH --nodes=x
#SBATCH --ntasks=x
#SBATCH --gres=gpu:x
#SBATCH --mem=16GB

export OMP_NUM_THREADS=1

WANDB_API_KEY=$YOUR_API_KEY

#For unconditioned generation
python train.py --run_name xxx --data_name trainingdataset_10properties --batch_size 192 --max_epochs 20

#For conditional generation
python train.py --run_name xxx --data_name trainingdataset_10properties --batch_size 192 --max_epochs 20 --props cond eox ce --num_props 3
