#!/bin/bash

#SBATCH --partition=longq
#SBATCH --nodes 1
#SBATCH --gres=gpu:xp:1

#module purge 
#module load gcc/5.2.0
module load easybuild
module load lang/Python/3.5.2-foss-2016b

#execute application
#python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'seed=27105828'  'db="file"' 'run_id="lstm_big_final_partTest_xp_100_15ep"'  'n_epoch=15' 'net_arch_layers = ["lstm","fc","output"]' 'net_arch.lstm.return_state=True' 'net_arch.lstm.return_seq=True'  'batch_size = 32' 'save_path = "/home/icha/sacred_models/"' 'tensorboard_dir = "/home/icha/sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"' 'save_mode="pickle"'


python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'seed=73467635' 'db="file"' 'run_id="lstm_small_final_partTest_50_moreLRP_moreTrain"' 'net_arch.lstm.n_units=20' 'net_arch_layers = ["lstm","fc","output"]' 'net_arch.lstm.return_state=True' 'net_arch.lstm.return_seq=True'  'batch_size = 32' 'save_path = "/home/icha/sacred_models/"' 'tensorboard_dir = "/home/icha/sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"' 'save_mode="pickle"' 'test_size=0.05'

#python3.5 /home/icha/tRustNN/src/dynamic_lstm_TF.py with 'seed=910361581'  'db="file"' 'run_id="lstm_2layers_with_inputJson"' 'net_arch.lstm.n_units=30' 'net_arch.lstm.return_seq=True' 'net_arch.lstm.return_state=True' 'net_arch.lstm2 = {"n_units":20, "activation":"tanh", "inner_activation":"sigmoid", "dropout":None, "bias":True, "weights_init":None, "forget_bias":1.0, "return_seq":True, "return_state":True, "initial_state":None, "dynamic":False, "trainable":True, "restore":True, "reuse":False, "scope":None, "name":"lstm2"}' 'net_arch_layers=["lstm","lstm2","fc","output"]' 'batch_size = 32' 'save_path = "/home/icha/sacred_models/"' 'tensorboard_dir = "/home/icha/sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"'


# exit
exit 0
