#!/bin/bash

#SBATCH --partition=longq
#SBATCH --nodes 1

#module purge 
module load gcc/5.2.0
#module load easybuild
#module load lang/Python/3.5.2-foss-2016b

#./home/icha/mongodb_3.4.4/bin/mongod --config mongodb.conf &

#execute application
python3 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"' 'net_arch.lstm.n_units=10'

# exit
exit 0
