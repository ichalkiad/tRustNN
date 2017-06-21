from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool
from data_format import get_data
from pprint import pprint
import numpy as np
from bokeh.models.widgets import Select, Slider
from sklearn.decomposition import PCA
import random
from bokeh.layouts import row, widgetbox, column
import clustering

def get_selections(keys):
    
    lstm_layers = [l for l in keys if "lstm" in l]
    select_layer = Select(title="LSTM layer", value="lstm", options=lstm_layers)

    gates = ["input_gate","forget_gate","output_gate"]
    select_gate = Select(title="Gate", value="input_gate", options=gates)

    return (select_layer,select_gate)


def get_lstmGate_source(lstm,gate,data):

    pca = PCA(n_components=2)
    x = data[lstm][gate]
    x_pca = pca.fit_transform(np.transpose(x))

    source = ColumnDataSource(data=dict(x=x_pca[:,0],y=x_pca[:,1],z=np.array([i for i in range(x_pca.shape[0])])))
    
    return source


def update_source(attrname, old, new):
    
    layer_value = selections[0].value
    gate_value  = selections[1].value

    pca = PCA(n_components=2)
   
    x = data[layer_value][gate_value]

    x_pca = pca.fit_transform(np.transpose(x))
    
    source.data = dict(x=x_pca[:,0],y=x_pca[:,1],z=np.array([i for i in range(x_pca.shape[0])]))

    #update clustering source
    algorithm = algorithm_select.value
    n_clusters = int(cluster_slider.value)
    x_cl, y_pred = clustering.clustering(x, algorithm, n_clusters)
    colors = [cl_spectral[i] for i in y_pred]
    cluster_plot.title.text = algorithm
    cluster_source.data = dict(colors=colors, x=x_cl[:, 0], y=x_cl[:, 1])

    
    
def update_clustering(attrname,old,new):

    algorithm = algorithm_select.value
    n_clusters = int(cluster_slider.value)

    x_cl, y_pred = clustering.clustering(data_cl, algorithm, n_clusters)
    colors = [cl_spectral[i] for i in y_pred]

    cluster_plot.title.text = algorithm
    cluster_source.data = dict(colors=colors, x=x_cl[:, 0], y=x_cl[:, 1])


    
#Get trained model parameters: weights and gate values
keys,data = get_data("/home/yannis/Desktop/tRustNN/bokeh_vis/data/model.json")

selections = get_selections(keys)
inputs = widgetbox(selections[0],selections[1])

source = get_lstmGate_source("lstm","input_gate",data)

hover = HoverTool()
hover.tooltips = [("Cell", "$index"),("(x,y)", "($x,$y)")]
hover.mode = 'mouse'

tools = "pan,wheel_zoom,box_zoom,reset,hover"
p = figure(title="PCA scatter plot",tools=tools)
p.scatter((source.data['x']), (source.data['y']), marker='circle', size=10, line_color="navy", fill_color="navy", alpha=0.5)

for attr in selections:
    attr.on_change('value', update_source)


#Clustering

data_cl = data[selections[0].value][selections[1].value]
cluster_source, cluster_plot, cl_spectral = clustering.apply_cluster(data_cl,'MiniBatchKMeans',2,tools=tools)

algorithm_select = Select(value='MiniBatchKMeans',title='Select algorithm:',width=200, options=clustering.get_cluster_algorithms())
cluster_slider = Slider(title="Number of clusters",value=2.0,start=2.0,end=10.0,step=1,width=400)

algorithm_select.on_change('value', update_clustering)
cluster_slider.on_change('value',  update_clustering)


curdoc().add_root(row(inputs, p, algorithm_select, cluster_slider, cluster_plot, width=400))
curdoc().title = "tRustNN"
