from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, show, curdoc
from bokeh.layouts import row, widgetbox, column, gridplot
from bokeh.models.widgets import Select, Slider
from sklearn.decomposition import PCA
from data_format import get_data
import dim_reduction
import numpy as np
import clustering
import random

def get_selections(keys):
    
    lstm_layers = [l for l in keys if "lstm" in l]
    select_layer = Select(title="LSTM layer", value="lstm", options=lstm_layers)

    gates = ["input_gate","forget_gate","output_gate"]
    select_gate = Select(title="Gate", value="input_gate", options=gates)

    return (select_layer,select_gate)


def get_clustering_selections(algorithms):

    algorithm_select = Select(value='MiniBatchKMeans',title='Select algorithm:',width=200, options=algorithms)
    cluster_slider = Slider(title="Number of clusters",value=2.0,start=2.0,end=10.0,step=1,width=400)

    return (algorithm_select,cluster_slider)


def get_projection_selections(algorithms):

    algorithm_select = Select(value='PCA',title='Select algorithm:',width=200, options=algorithms)
    knn_slider = Slider(title="Number of neighbors",value=5.0,start=5.0,end=30.0,step=1,width=200)
    dim_slider = Slider(title="Number of dimensions",value=2,start=2,end=3,step=1,width=200)
    
    return (algorithm_select,knn_slider,dim_slider)



def update_source(attrname, old, new):
    
    layer_value = gate_selections[0].value
    gate_value  = gate_selections[1].value

    x = data[layer_value][gate_value]

    #update dimension reduction source
    algorithm = projection_selections[0].value
    knn = int(projection_selections[1].value)
    dimensions = int(projection_selections[2].value)
    x_pr = dim_reduction.project(x, algorithm, knn, dimensions, labels)
    project_plot.title.text = algorithm

    #update clustering 
    algorithm = clustering_selections[0].value
    n_clusters = int(clustering_selections[1].value)
    cluster_labels, colors, cl_spectral = clustering.apply_cluster(x,algorithm,n_clusters)

    proj_source.data = dict(x=x_pr[:, 0], y=x_pr[:, 1], z=colors)




    

    
#Get trained model parameters: weights and gate values
keys,data = get_data("/home/yannis/Desktop/tRustNN/bokeh_vis/data/model.json")


#LSTM gates
gate_selections = get_selections(keys)
gate_inputs = widgetbox(gate_selections[0],gate_selections[1])
#Clustering
clustering_selections = get_clustering_selections(clustering.get_cluster_algorithms())
clustering_inputs = widgetbox(clustering_selections[0],clustering_selections[1])

hover = HoverTool()
hover.tooltips = [("Cell", "$index"),("(x,y)", "($x,$y)")]
hover.mode = 'mouse'
tools = "pan,wheel_zoom,box_zoom,reset,hover"

#Dimensionality reduction
labels = None # LOAD GROUND TRUTH OR NET-ASSIGNED LABELS??
data_pr = data[gate_selections[0].value][gate_selections[1].value]
X = dim_reduction.project(data_pr, 'PCA', n_neighbors=10, n_components=2, labels=labels)
cluster_labels, colors, cl_spectral = clustering.apply_cluster(data_pr,'MiniBatchKMeans',2)
proj_source = ColumnDataSource(dict(x=X[:,0],y=X[:,1],z=colors))

projection_selections = get_projection_selections(dim_reduction.get_dimReduction_algorithms())
projection_inputs = widgetbox(projection_selections[0],projection_selections[1],projection_selections[2])

project_plot = figure(title=projection_selections[0].value,tools=tools)
project_plot.scatter('x', 'y', marker='circle', size=10, fill_color='z', alpha=0.5, source=proj_source)


#Layout
for attr in gate_selections:
    attr.on_change('value', update_source)
for attr in projection_selections:
    attr.on_change('value', update_source)
for attr in clustering_selections:
    attr.on_change('value', update_source)


#row(gate_inputs, projection_inputs, project_plot, clustering_inputs, cluster_plot, width=400)
gp = gridplot([[project_plot, None],[row(gate_inputs, projection_inputs,clustering_inputs)]])
curdoc().add_root(gp)
curdoc().title = "tRustNN"
