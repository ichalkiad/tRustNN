#!/bin/bash

SBATCH --partition=longq
SBATCH --nodes 2
SBATCH --mail-user=ic14@hw.ac.uk
SBATCH --mail-type=ALL 

#module purge 
#module load gcc/5.2.0

./home/icha/mongodb_3.4.4/bin/mongod --config mongodb.conf &

#execute application
python3 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"'

# exit
exit 0
