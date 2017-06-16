#!/bin/bash

#SBATCH --partition=longq
#SBATCH --nodes 1
#SBATCH --gres=gpu:1

#module purge 
#module load gcc/5.2.0
module load easybuild
module load lang/Python/3.5.2-foss-2016b

#execute application
python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"' 'run_id="baseline"'  'net_arch_layers = ["lstm","fc","output"]' 'net_arch.lstm.return_state=True' 'batch_size = 32' 'save_path = "/home/icha/sacred_models/"' 'tensorboard_dir = "/home/icha/sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"'


python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"' 'run_id="lstm_small"' 'net_arch.lstm.n_units=20' 'net_arch_layers = ["lstm","fc","output"]' 'net_arch.lstm.return_state=True'  'batch_size = 32' 'save_path = "/home/icha/sacred_models/"' 'tensorboard_dir = "/home/icha/sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"'

python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'db="file"' 'run_id="lstm_2layers"' 'net_arch.lstm.n_units=30' 'net_arch.lstm.return_seq=True' 'net_arch.lstm.return_state=True' 'net_arch.lstm2 = {"n_units":20, "activation":"tanh", "inner_activation":"sigmoid", "dropout":None, "bias":True, "weights_init":None, "forget_bias":1.0, "return_seq":False, "return_state":True, "initial_state":None, "dynamic":True, "trainable":True, "restore":True, "reuse":False, "scope":None, "name":"lstm2"}' 'net_arch_layers=["lstm","lstm2","fc","output"]' 'batch_size = 32' 'save_path = "/home/icha/sacred_models/"' 'tensorboard_dir = "/home/icha/sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"'


# exit
exit 0
