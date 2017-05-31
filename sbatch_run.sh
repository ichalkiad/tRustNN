#!/bin/bash

SBATCH --partition=defq
#SBATCH --nodes 2
SBATCH --mail-user=ic14@hw.ac.uk
SBATCH --mail-type=ALL 

#module purge 
#module load gcc/5.2.0

#execute application
python3 dynamic_lstm_TF.py

# exit
exit 0
