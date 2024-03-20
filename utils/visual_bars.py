import dgl
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np 
import pickle as pkl
import argparse
import os
from collections import defaultdict

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def run():
   
    viridis = mpl.colormaps['viridis'].resampled(8)

    src = [i*2 for i in range(8)]
    dst = [ i+1 for i in src]
    
    with open('etypes.pkl', 'rb') as f:
        etypes = pkl.load(f)
    print(etypes)
    etypes = list(etypes)
    print(len(etypes))

    edge_types = []
    for et in etypes:
        edge_types.append(et)
        edge_types.append(et)

    g = dgl.graph((src, dst))
    G = dgl.to_networkx(g)
    for i in range(8*2):
        G.nodes[i]['type'] = edge_types[i]
    
    pos = nx.spring_layout(G)
    # Draw nodes and labels 
    nx.draw_networkx_nodes(G, pos,  node_size=1) 
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'type'), font_size=10) 
    # Draw edges with weights 
    for i in range(8):
        color = viridis(i/8)
        nx.draw_networkx_edges(G, pos, edgelist=[(2*i,2*i+1)], edge_color=color) 

    #nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, '_TYPE')) 
    #nx.draw_networkx_edges(G, pos) 
    # Show graph 
    plt.axis('off') 
    plt.savefig("path.pdf")

run()