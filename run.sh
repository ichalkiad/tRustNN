python3 src/dynamic_lstm_TF.py with 'db="file"' 'run_id="lstm_2layers"' 'net_arch.lstm.n_units=30' 'net_arch.lstm.return_seq=True' 'net_arch.lstm.return_state=True' 'net_arch.lstm2 = {"n_units":20, "activation":"tanh", "inner_activation":"sigmoid", "dropout":None, "bias":True, "weights_init":None, "forget_bias":1.0, "return_seq":True, "return_state":True, "initial_state":None, "dynamic":False, "trainable":True, "restore":True, "reuse":False, "scope":None, "name":"lstm2"}' 'net_arch_layers=["lstm","lstm2","fc","output"]' 'batch_size = 1' 'save_path = "./sacred_models/"' 'tensorboard_dir = "./sacred_models/tf_logs/"' 'embedding_dim = 300' 'internals = "all"'


#python3 src/dynamic_lstm_TF.py with 'db="file"' 'run_id="test"' 'net_arch.lstm.n_units=20'


#'net_arch.lstm.return_seq=True' 'net_arch.lstm2 = {"n_units":20, "activation":"tanh", "inner_activation":"sigmoid", "dropout":None, "bias":True, "weights_init":None, "forget_bias":1.0, "return_seq":False, "return_state":False, "initial_state":None, "dynamic":False, "trainable":True, "restore":True, "reuse":False, "scope":None, "name":"lstm2"}' 'net_arch_layers=["lstm","lstm2","fc","output"]'

