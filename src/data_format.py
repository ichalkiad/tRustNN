import numpy as np
import json
import pickle
from collections import defaultdict

def return_keys(json_file):

    if "json" in json_file:
        with open(json_file) as data_file:    
            data = json.load(data_file)
    elif "pickle" in json_file:
        with open(json_file,"rb") as data_file:    
            data = pickle.load(data_file)

    return data.keys()


def return_JSONcontent(keys,json_file):

    if "json" in json_file:
        with open(json_file) as data_file:    
            data = json.load(data_file)
    elif "pickle" in json_file:
        with open(json_file,"rb") as data_file:    
            data = pickle.load(data_file)

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


def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
        
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)
