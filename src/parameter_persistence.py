import tensorflow as tf
import tflearn
import json

def export_serial_model(model,layer_names,save_dir):

    network = dict()
    
    for l in layer_names:
        if l=="output":
            break
        layer = tflearn.variables.get_layer_variables_by_name(l)
        network[l] = dict()
        data_arr = model.get_weights(layer[0])
        network[l]['W'] = data_arr.tolist()
        data_arr = model.get_weights(layer[1])
        network[l]['b'] = data_arr.tolist()


    with open(save_dir+"model.json", 'w') as f:
         json.dump(network, f)
   
   
