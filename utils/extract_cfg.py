import ast
import pandas as pd
import pickle as pkl
import argparse
import os
from tqdm import tqdm
import json

class ControlFlowNode:
    def __init__(self, node_id, kind, name):
        self.id = node_id
        self.kind = kind
        self.name = name

class ControlFlowGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.current_id = 0
        self.nodes = {}
        self.edges = []
        self.last_id = None
        self.function_defs = {}

    def add_node(self, node):
        node_id = self.current_id
        self.current_id += 1
        node_kind = type(node).__name__
        node_name = getattr(node, 'name', None) or getattr(node, 'arg', None) or node_kind
        self.nodes[node_id] = ControlFlowNode(node_id, node_kind, node_name)
        if self.last_id is not None:
            self.edges.append({'between': [self.last_id, node_id], 'edgeType': 'Next'})
        self.last_id = node_id
        return node_id

    def visit_FunctionDef(self, node):
        func_id = self.add_node(node)
        self.generic_visit(node)
        self.function_defs[node.name] = (func_id, self.last_id)

    def visit_Call(self, node):
        call_id = self.add_node(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.function_defs:
            func_def_id, func_end_id = self.function_defs[node.func.id]
            self.edges.append({'between': [call_id, func_def_id], 'edgeType': 'CallNext'})
            self.edges.append({'between': [func_end_id, self.current_id], 'edgeType': 'ReturnNext'})

        self.generic_visit(node)

    def visit_If(self, node):
        if_id = self.add_node(node)
        last_id_before_if = if_id
        self.generic_visit(node.test)
        if node.body:
            first_body_id = self.add_node(node.body[0])
            self.edges.append({'between': [last_id_before_if, self.last_id], 'edgeType': 'PosNext'})
            for child in node.body[1:]:
                self.visit(child)
            last_body_id = self.last_id
        if node.orelse:
            first_orelse_id = self.add_node(node.orelse[0])
            self.edges.append({'between': [last_id_before_if, self.last_id], 'edgeType': 'NegNext'})
            for child in node.orelse[1:]:
                self.visit(child)
            last_orelse_id = self.last_id
        else:
            self.edges.append({'between': [last_id_before_if, self.current_id], 'edgeType': 'NegNext'})
        if node.orelse:
            self.last_id = last_orelse_id
        elif node.body:
            self.last_id = last_body_id
        else:
            self.last_id = last_id_before_if

    def visit_For(self, node):
        for_id = self.add_node(node)
        self.generic_visit(node.iter)
        if node.body:
            first_stmt_id = self.add_node(node.body[0])
            self.edges.append({'betweeen': [for_id, first_stmt_id], 'edgeType': 'PosNext'})
            for child in node.body[1:]:
                self.visit(child)
            self.edges.append({'between': [self.last_id, for_id], 'edgeType': 'IterJump'})
        self.edges.append({'between': [for_id, self.current_id], 'edgeType': 'NegNext'})
        self.last_id = for_id

    def visit_While(self, node):
        for_id = self.add_node(node)
        self.generic_visit(node.test)
        if node.body:
            first_stmt_id = self.add_node(node.body[0])
            self.edges.append({'betweeen': [for_id, first_stmt_id], 'edgeType': 'PosNext'})
            for child in node.body[1:]:
                self.visit(child)
            self.edges.append({'between': [self.last_id, for_id], 'edgeType': 'IterJump'})
        self.edges.append({'between': [for_id, self.current_id], 'edgeType': 'NegNext'})
        self.last_id = for_id

    def generic_visit(self, node):
        if hasattr(node, 'body') and isinstance(node.body, list):
            for child in node.body:
                self.visit(child)
        else:
            self.add_node(node)
            super().generic_visit(node)

def build_cfg_from_code(code):
    try:
        parsed_ast = ast.parse(code)
        cfg_builder = ControlFlowGraphBuilder()
        cfg_builder.visit(parsed_ast)
    
        node_list = [{'ID': node.id, 'kind': node.kind, 'name': node.name} 
                     for node_id, node in cfg_builder.nodes.items()]
        edge_list = [edge for idx, edge in enumerate(cfg_builder.edges)]
    
        return node_list, edge_list
    except SyntaxError:
        print("Syntax error encountered. Skipping this code segment.")
        return None

def extract_cfgs():
    with open('/home/knhdu/ext0/CodeRAG/data/LeetcodeData/codes.pkl', 'rb') as f:
        python_codes = pkl.load(f)
    
    available_codes = []
    for i, code in enumerate(python_codes):
        cfg = build_cfg_from_code(code)
        if cfg is not None:
            available_codes.append(code)
            nodes, edges = cfg
            graph = [nodes, edges, "code_"+str(i)]   
            with open("/home/knhdu/ext0/CodeRAG/data/LeetcodeData/graph/code_"+str(i)+".pkl", "wb") as file:
                 pkl.dump(graph, file, protocol=4)
    
    with open("/home/knhdu/ext0/CodeRAG/data/LeetcodeData/available_codes.pkl", "wb") as file:
         pkl.dump(available_codes, file, protocol=4)
    


#if __name__ == "__main__":
    '''df = pd.read_parquet("dataset.parquet")
    python_codes = df['output'].tolist()
    print(len(python_codes))
    assert 0
    available_codes = []
    for i, code in enumerate(python_codes):
        cfg = build_cfg_from_code(code)
        if cfg is not None:
            available_codes.append(code)
            nodes, edges = cfg
            graph = [nodes, edges, "code_"+str(i)]   
            with open("graph/code_"+str(i)+".pkl", "wb") as file:
                 pickle.dump(graph, file, protocol=4)

    with open("available_codes.pkl", "wb") as file:
         pickle.dump(available_codes, file, protocol=4)'''
    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--path", default="../output", help="data path")
    parser.add_argument("--file", default="samples.jsonl", help="data path")
    parser.add_argument("--output", default="../output/OneRoundRes", help="data path")
    
    args = parser.parse_args()
    

    def extract_code(line):
        code_name = line['task_id'][7:]
        code_name = 'Python_' + code_name
        code_content = line['declaration'] + line['generation']
        return code_content

    def main(path, result_file):
        cnt = 0
        with open(os.path.join(path, result_file), 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)
                code = extract_code(line)
                cfg = build_cfg_from_code(code)
                if cfg is not None:
                    cnt += 1
        print(cnt)

    main(args.path, args.file)
        