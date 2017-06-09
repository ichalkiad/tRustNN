#!/bin/bash

#SBATCH --partition=longq
#SBATCH --nodes 1

#module purge 
#module load gcc/5.2.0
module load easybuild
module load lang/Python/3.5.2-foss-2016b

#execute application
python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"' 'run_id="baseline"' 

python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"' 'run_id="base_20LSTM"' 'net_arch.lstm.n_units=20'

python3 src/dynamic_lstm_TF.py with 'db="file"' 'run_id="2Layers_50-20LSTM"' 'net_arch.lstm.n_units=50' 'net_arch.lstm.return_seq=True' 'net_arch.lstm2 = {"n_units":30, "activation":"tanh", "inner_activation":"sigmoid", \
"dropout":None, "bias":True, "weights_init":None, "forget_bias":1.0, "return_seq":False, "return_state":False, "initial_state":None, "dynamic":False, "trainable":True, "restore":True, "reuse":False, "scope":None\
, "name":"lstm2"}' 'net_arch_layers=["lstm","lstm2","fc","output"]'


# exit
exit 0
