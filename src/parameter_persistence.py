import tensorflow as tf
import tflearn
import json


def split_lstm_output(T,name):

    (o,state) = T
       
    return o

def export_serial_model(model,layer_names,save_dir):

    network = dict()
    
    for l in layer_names:
        if "output" in l:
            break
        network[l] = dict()
        layer = tflearn.variables.get_layer_variables_by_name(l)
        if "lstm" in l:
            data_arr = model.get_weights(layer[0])
            #Each of the following matrices: (embedding_size + hidden_layer_size) x LSTM_cell_number , first part coming from concatenating new_input + output from previous cell
            input, new_input, forget, output = tf.split(layer[0], num_or_size_splits=4, axis=1)
            input_arr     = model.session.run(input).tolist()
            network[l]['input_gate'] = input_arr
            # New_input_arr is needed if we want to restore network's exact state when we stopped training
            new_input_arr = model.session.run(new_input).tolist()
            network[l]['new_input'] = new_input_arr
            forget_arr    = model.session.run(forget).tolist()
            network[l]['forget_gate'] = forget_arr
            output_arr    = model.session.run(output).tolist()
            network[l]['output_gate'] = output_arr
        if "fc" in l:
            data_arr = model.get_weights(layer[0])
            network[l]['W'] = data_arr.tolist()
        
        data_arr = model.get_weights(layer[1])
        network[l]['b'] = data_arr.tolist()


    with open(save_dir+"model.json", 'w') as f:
         json.dump(network, f)
   
   
