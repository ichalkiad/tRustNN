import numpy as np
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import figure
from bokeh.palettes import Spectral6

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def clustering(X, algorithm, n_clusters):
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Generate the new colors:
    if algorithm=='MiniBatchKMeans':
        model = cluster.MiniBatchKMeans(n_clusters=n_clusters)

    elif algorithm=='Birch':
        model = cluster.Birch(n_clusters=n_clusters)

    elif algorithm=='DBSCAN':
        model = cluster.DBSCAN(eps=.2)

    elif algorithm=='AffinityPropagation':
        model = cluster.AffinityPropagation(damping=.9,
                                            preference=-200)

    elif algorithm=='MeanShift':
        model = cluster.MeanShift(bandwidth=bandwidth,
                                  bin_seeding=True)

    elif algorithm=='SpectralClustering':
        model = cluster.SpectralClustering(n_clusters=n_clusters,
                                           eigen_solver='arpack',
                                           affinity="nearest_neighbors")

    elif algorithm=='Ward':
        model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage='ward',
                                                connectivity=connectivity)

    elif algorithm=='AgglomerativeClustering':
        model = cluster.AgglomerativeClustering(linkage="average",
                                                affinity="cityblock",
                                                n_clusters=n_clusters,
                                                connectivity=connectivity)

    model.fit(X)

    if hasattr(model, 'labels_'):
            y_pred = model.labels_.astype(np.int)
    else:
            y_pred = model.predict(X)

    return X, y_pred



def apply_cluster(data,algorithm,n_clusters):

    spectral = np.hstack([Spectral6] * 20)

    x_cl, y_pred = clustering(data, algorithm, n_clusters)
    colors = [spectral[i] for i in y_pred]

    source = ColumnDataSource(data=dict(x=x_cl[:, 0], y=x_cl[:, 1], colors=colors))

    return source, colors, spectral


def get_cluster_algorithms():

    return [
        'MiniBatchKMeans',
        'AffinityPropagation',
        'MeanShift',
        'SpectralClustering',
        'Ward',
        'AgglomerativeClustering',
        'DBSCAN',
        'Birch'
    ]


    

