import numpy as np
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import Birch
from matplotlib.patches import Rectangle

def generate_graph_from_gaze_df(gaze_df, img_width, img_height, threshold=0.1, branching_factor=150, n_clusters=None):
    # remove unused columns and  null values.
    gaze_df = gaze_df[['x_position', 'y_position']]
    gaze_df = gaze_df[~(gaze_df['x_position'].isna() | gaze_df['x_position'].isna())]
    gaze_df = gaze_df[gaze_df['x_position'].between(0, img_width)]
    gaze_df = gaze_df[gaze_df['y_position'].between(0, img_height)]

    # create gaze points
    gaze_points = np.array([(x,y) for x, y in zip(gaze_df['x_position'], gaze_df['y_position'])])

    # set the centre of the cxr as (0,0) and normalise x,y.
    centre_x = img_width/2
    centre_y = img_height /2
    norm_gaze_points = (gaze_points - np.array([centre_x, centre_y])) / np.array([centre_x, centre_y])

    # Initialize the BIRCH algorithm
    birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)

    # Fit the BIRCH algorithm to the gaze point data
    birch.fit(norm_gaze_points)

    # Generate the clustering output
    subcluster_labels = birch.predict(norm_gaze_points)

    # assign cluster labels for each gaze point
    gaze_df['cluster'] = subcluster_labels.astype(int)

    """
    create adjacency matrix
    """
    # create an directed adjacency matrix first.
    num_clusters = len(birch.subcluster_labels_)
    directed_adjacency =  np.zeros((num_clusters, num_clusters)).astype(int)

    # follow the setup in the paper to perform edge weights assignment.
    # 1. all the vertices pertaining to each cluster are removed except the centroid of the cluster,
    # 2. all the edges that were connecting vertices from different clusters now connect the corresponding centroids, and
    # 3. all the edges that were connecting vertices inside each cluster are modeled as self loops on the centroid.
    ### (paper: https://arxiv.org/abs/1802.06260)

    for i in range(len(gaze_df)-1):
        u_cluster = gaze_df.iloc[i]['cluster'] 
        v_cluster = gaze_df.iloc[i+1]['cluster'] 
        directed_adjacency[int(u_cluster), int(v_cluster)] += 1 
        
    # then, transform the adjacency matrix to undirected.    
    undirected_adjacency = directed_adjacency + directed_adjacency.T
    for i in range(len(undirected_adjacency)):
        undirected_adjacency[i, i] /=2

    """
    generate graph.
    """

    norm_centroid_x = birch.subcluster_centers_[:, 0]
    norm_centroid_y = birch.subcluster_centers_[:, 1]

    G = nx.Graph()
    # Add nodes to the graph and set their positions
    for (i, x, y) in zip(birch.subcluster_labels_, norm_centroid_x, norm_centroid_y):
        N = len(gaze_df[gaze_df['cluster'] == i])
        C = undirected_adjacency[i,i] + 1
        G.add_node(i, pos=(x,y), N=N, C=C)
    
    # Add edges to the graph and set their weights
    for i in range(len(undirected_adjacency)):
        for j in range(len(undirected_adjacency[i])):
            if undirected_adjacency[i][j] > 0:
                G.add_edge(i, j, weight=undirected_adjacency[i][j])


    return G
   

def visualise_graph_with_weights(G, figsize=(25,15)):
    plt.figure(figsize=figsize) 
    plt.gca().invert_yaxis()
    pos = nx.get_node_attributes(G, 'pos')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_widths = [np.log10(w + 1) for (_, _, w) in G.edges(data='weight')]
    nx.draw_networkx(G, pos=pos, with_labels=True, width=edge_widths)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.show()


# def visualise_graph(G, figsize=(25,15)):
#     plt.figure(figsize=figsize) 
#     plt.gca().invert_yaxis()
#     pos = nx.get_node_attributes(G, 'pos')
#     nx.draw_networkx(G, pos=pos, with_labels=True)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
#     plt.show()