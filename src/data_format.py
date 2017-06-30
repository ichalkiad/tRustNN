import numpy as np
import json


def return_keys(json_file):

    with open(json_file) as data_file:    
        data = json.load(data_file)

    return data.keys()


def return_JSONcontent(keys,json_file):

    with open(json_file) as data_file:    
        data = json.load(data_file)

    for k in keys:
        if isinstance(data[k],dict):
            for kk in data[k].keys():
                data[k][kk] = np.array(data[k][kk])
        else:
            data[k] = np.array(data[k])

    return data

                           
def get_data(json):

    keys = return_keys(json)
    data = return_JSONcontent(keys,json)

    return keys,data
                           
