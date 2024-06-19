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


def build_graph(path, label_map):
    with open(path, 'rb') as f:
        ndata, edata, name = pkl.load(f)
    
    edges_src = defaultdict(list)
    edges_dst = defaultdict(list)
    etypes = set()
    max_nid = 0

    for edge_info in edata:
        src, dst = edge_info.get('between', [-1, -1])
        if src == -1:
            src, dst = edge_info['betweeen']
        if src is None or dst is None:
            return None
        etype = edge_info['edgeType']
        etypes.add(etype)
        edges_src[etype].append(src)
        edges_dst[etype].append(dst)
        max_nid = max(max_nid, src, dst)
    
    data_dict = {
        ('node', etype, 'node'): (edges_src[etype], edges_dst[etype])
        for etype in list(etypes)
    }
    #print(data_dict)
    num_nodes_dict = {
        'node': len(ndata)
    }

    if max_nid >= len(ndata):
        return None

    if data_dict:
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        #g = dgl.heterograph(data_dict)
    else:   
        return None, None, None

    viridis = mpl.colormaps['viridis'].resampled(len(g.etypes))

    if label_map == None:
        label = None
    else:
        label = label_map[name]
    
    etypes = np.array(g.etypes)
    with open('etypes.pkl', 'wb') as f:
        pkl.dump(etypes, f)

    homo_g = dgl.to_homogeneous(g)
    
    G = dgl.to_networkx(homo_g, edge_attrs=['_TYPE'])
    for i, node_info in enumerate(ndata):
        G.nodes[i]['type'] = node_info['type']
        G.nodes[i]['kind'] = node_info['kind']
        G.nodes[i]['name'] = node_info['name']
    
    with open('cpp_nx.pkl', 'wb') as f:
        pkl.dump(G, f)
    
    '''nx_edges = G.edges(data=True)

    pos = nx.spring_layout(G)

    ec = [e for e in nx_edges]
    edge_weight = nx.get_edge_attributes(G, '_TYPE')

    options = {"node_size": 1000,
                "alpha": 0.3,
                "font_size": 12,
                "edge_color": ec,
                "width": 4,
                "edge_cmap": plt.cm.Reds,
                "edge_vmin": 0,
                "edge_vmax": 1,
                "connectionstyle":"arc3,rad=0.1"}

    nx.draw(G, pos, with_labels=True, node_color='b', **options)
    '''

    nx_edges = G.edges(data=True)
    edge_types = defaultdict(list)
    for edge in nx_edges:
        edge_types[edge[2]['_TYPE']].append((edge[0], edge[1]))
    
    print(G)
    
    pos = nx.kamada_kawai_layout(G, scale=1)
    # Draw nodes and labels 
    nx.draw_networkx_nodes(G, pos, node_color = 'pink') 
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'type'), font_size=8) 
    # Draw edges with weights 
    for i, edg in edge_types.items():
        color = viridis(i.item()/len(g.etypes))
        nx.draw_networkx_edges(G, pos, edgelist=edg, edge_color=color) 

    #nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, '_TYPE')) 
    #nx.draw_networkx_edges(G, pos) 
    # Show graph 
    plt.axis('off') 
    plt.savefig("path.pdf")

def process_data(args, data_path, label_path, output_path):
    graph_path = os.path.join(data_path, 'graph')
    graph_files = [os.path.join(graph_path, file) for file in os.listdir(graph_path)]
    
    label_map = None
    
    '''file = graph_files[args.i]
    print(file)'''
    build_graph('../data/humaneval-x/cpp/data/graph/CPP_48.pkl', label_map)
        

     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--datapath", default="../data/CPPData/CodeT5/LongCodes", help="data path")
    parser.add_argument("--output", default="../data/CPPData/CodeT5/LongCodes", help="output path")
    parser.add_argument("--i", type=int)

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)      

    process_data(args,args.datapath, label_path = None, output_path=args.output)





