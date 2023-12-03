import dgl
import pickle as pkl
import os
import sys
sys.path.append("..")
from tqdm import tqdm

def descibe_graphs(graph_path):
    with open(os.path.join(graph_path, 'graphs.pkl'), 'rb') as f:
        graphs = pkl.load(f)
    graph_list = []
    for g in tqdm(graphs):
        des = 'The abstract syntax and control flow of a relative code block could be represented as follows: This is a graph. '
        edge_types = g.etypes
        edges = {etype: g.edges(etype=etype) for etype in g.etypes}
        edges = {key: (value[0].tolist(), value[1].tolist()) for key, value in edges.items()}
        type_des = 'The types of the edges of the graph include: ' + str(edge_types)
        #edge_des = 'The edges of each edge type are given as a 2-tuple of lists (U,V), representing the source and destination nodes of all edges. For each i, (U[i], V[i]) forms an edge. The edges of each edge type are: ' + str(edges)
        des = des + type_des + '.\n' 
        #+ edge_des + '.\n'
        graph_list.append(des)
    with open(os.path.join(graph_path, 'graph_des.pkl'), 'wb') as f:
         pkl.dump(graph_list, f)

descibe_graphs('../data/Cgraphs')
