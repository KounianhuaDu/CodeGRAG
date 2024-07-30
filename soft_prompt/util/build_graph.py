import dgl
import pickle as pkl
from collections import defaultdict
import argparse
from tqdm import tqdm
import json
import os
import torch
from unixcoder import UniXcoder
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "/home/jzchen/ML/Code/models/codet5p-110m-embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UniXcoder("/home/jzchen/ML/codenew/Code/model_weights/unixcoder-base-nine")
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)


def build_graph(path):
    with open(path, "rb") as f:
        ndata, edata = pkl.load(f)

    edges_src = defaultdict(list)
    edges_dst = defaultdict(list)
    etypes = set()
    max_nid = 0

    for edge_info in edata:
        src, dst = edge_info.get("between", [-1, -1])
        if src == -1:
            src, dst = edge_info["betweeen"]
        if src is None or dst is None:
            return None
        etype = edge_info["edgeType"]
        etypes.add(etype)
        edges_src[etype].append(src)
        edges_dst[etype].append(dst)
        max_nid = max(max_nid, src, dst)

    data_dict = {
        ("node", etype, "node"): (edges_src[etype], edges_dst[etype])
        for etype in list(etypes)
    }
    num_nodes_dict = {"node": len(ndata)}

    if max_nid >= len(ndata):
        return None

    if data_dict:
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    else:
        return None

    node_kinds = []
    node_types = []
    edge_types = defaultdict(list)
    for node_info in ndata:
        node_kinds.append(node_info["kind"])
        node_types.append(node_info["name"])
    for edge_info in edata:
        edge_types[edge_info["edgeType"]].append(edge_info["edgeType"])
    n_kinds = tokenizer(
        node_kinds, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        n_kinds = model(**n_kinds)
    n_types = tokenizer(
        node_types, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        n_types = model(**n_types)
    for et, ev in edge_types.items():
        a = tokenizer(ev, padding="longest", truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            a = model(**a)
        edge_types[et] = a
    g.nodes["node"].data["kind"] = n_kinds.cpu()
    g.nodes["node"].data["type"] = n_types.cpu()
    for et, ev in edge_types.items():
        g.edges[et].data["h"] = ev.cpu()
    return g


def get_code(file_name):
    with open(
        os.path.join(
            "/home/jzchen/ML/AfterCodegrag/raw_data/APPS/train",
            file_name.split("_")[0],
            "solutions.json",
        ),
        "r",
    ) as f:
        return json.load(f)[int(file_name.split("_")[1][: -len(".pkl")])]


def process_data(data_path, output_path):
    graph_lists = os.listdir(data_path)
    graphs = []
    codes = []
    for file in tqdm(graph_lists):
        code = get_code(file)
        g = build_graph(os.path.join(data_path, file))
        if not g:
            continue
        graphs.append(g)
        codes.append(code)
    with open(os.path.join(output_path, "graphs.pkl"), "wb") as f:
        pkl.dump(graphs, f)
    with open(os.path.join(output_path, "codes.pkl"), "wb") as f:
        pkl.dump(codes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using different models to generate function"
    )

    parser.add_argument(
        "--datapath",
        default="/home/jzchen/ML/AfterCodegrag/raw_data/APPS/cfg",
        help="data path",
    )
    parser.add_argument(
        "--output",
        default="/home/jzchen/ML/AfterCodegrag/raw_data/APPS/dgl",
        help="output path",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    process_data(args.datapath, output_path=args.output)
