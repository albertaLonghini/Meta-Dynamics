
#! /bin/bash

for split in 0.1 0.5; do
  for n in 5 20; do
    sbatch --export=MAML=0,TS=4,NN=128,NL=1,N=$n,K=1,LR=0.0001,SPLIT=$split,DLO=1 script.sbatch
    sleep 1
    sbatch --export=MAML=1,TS=4,NN=128,NL=1,N=$n,K=20,LR=0.0001,SPLIT=$split,DLO=1 script.sbatch
    sleep 1
  done
done
