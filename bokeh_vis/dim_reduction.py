import numpy as np
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import figure

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

"""
    elif algorithm=="LDA": # Needs labels
        X.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
        X = discriminant_analysis.LinearDiscriminantAnalysis(n_components).fit_transform(X, labels)
"""
def project(X, algorithm, n_neighbors,n_components, labels=None):

    X = np.transpose(X)
    
    if algorithm=="LSA":
        X = decomposition.TruncatedSVD(n_components).fit_transform(X)
    elif algorithm=="PCA":
        X = decomposition.PCA(n_components).fit_transform(X)
    elif algorithm=="ISOMAP":
        X = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    elif algorithm=="LLE":
        lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
        X = lle.fit_transform(X)
    elif algorithm=="MDS":
        mds = manifold.MDS(n_components, n_init=1, max_iter=100)
        X = mds.fit_transform(X)
    elif algorithm=="tSNE":
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X = tsne.fit_transform(X)

    return X



def dim_reduce(data, algorithm, n_neighbors, n_components, labels=None):

    data_pr = project(data, algorithm, n_neighbors,n_components, labels)
    source = ColumnDataSource(data=dict(x=data_pr[:, 0], y=data_pr[:, 1]))

    return source


def get_dimReduction_algorithms():

    """
    'LDA',
    """
    return ['LSA',
            'PCA',
            'ISOMAP',
            'LLE',
            'MDS',
            'tSNE'
           ]


    

