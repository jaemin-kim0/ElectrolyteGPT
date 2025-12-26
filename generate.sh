#!/bin/bash
#SBATCH -t 36:00:00
#SBATCH --exclusive
#SBATCH --partition=xxx
#SBATCH --nodes=x
#SBATCH --ntasks=x
#SBATCH --gres=gpu:x
#SBATCH --mem=16GB

export OMP_NUM_THREADS=1

#Unconditioned generation
python generate.py --model_weight xxx.pt --data_name trainingdataset_10properties --csv_name unconditioned_xxx --gen_size 10000 --batch_size 192 --vocab_size 82 --block_size 190

#Conditional generation
python generate.py --model_weight xxx.pt --data_name trainingdataset_10properties --csv_name 3prop_xxx --gen_size 10000 --batch_size 192 --vocab_size 82 --block_size 190 --props cond eox ce
