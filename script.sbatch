#!/usr/bin/env bash
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constrain='smaug|shelob'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_torch

python3 /Midgard/home/areichlin/Robert/main.py --maml ${MAML} --t_steps ${TS} --n_neurons ${NN} --n_layers ${NL} --N ${N} --K ${K} --lr ${LR} --test_split ${SPLIT} --dlo_only ${DLO}