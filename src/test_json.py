import IMDB_dataset.imdb_preprocess as imdb_pre
from IMDB_dataset.textData import filenames_test
import json

data = imdb_pre.get_test_json(filenames_test)

with open("./bokeh_vis/test_data_input.json", "w") as f:
     json.dump(data, f)

