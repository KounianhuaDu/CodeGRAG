import ast
import os
import pickle as pkl
import os
from tqdm import tqdm
import argparse

class DataFlowNode:
    def __init__(self, node_id, name, kind):
        self.id = node_id
        self.name = name
        self.kind = kind

class DataFlowEdge:
    def __init__(self, src, dest, edge_type):
        self.between = [src, dest]
        self.edgeType = edge_type

class DataFlowGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.current_id = 0
        self.nodes = {}
        self.edges = []
        self.temp_counter = 0
        self.function_defs = {}
        self.call_to_return_map = {}

    def new_temp(self):
        self.temp_counter += 1
        return f"temp_{self.temp_counter}"

    def add_node(self, name, kind):
        node_id = self.current_id
        self.current_id += 1
        self.nodes[node_id] = DataFlowNode(node_id, name, kind)
        return node_id

    def add_edge(self, src, dest, edge_type):
        self.edges.append(DataFlowEdge(src, dest, edge_type))

    def visit_Name(self, node):
        return self.add_node(node.id, 'Name')

    def visit_Num(self, node):
        return self.add_node(str(node.n), 'Num')

    def visit_BinOp(self, node):
        left_id = self.visit(node.left)
        right_id = self.visit(node.right)
        result_id = self.add_node(self.new_temp(), 'temp')
        op_type = type(node.op).__name__
        self.add_edge(left_id, result_id, op_type)
        self.add_edge(right_id, result_id, op_type)
        return result_id

    def visit_Assign(self, node):
        if isinstance(node.value, (ast.Tuple, ast.List)) and len(node.value.elts) == len(node.targets):
            for target, value in zip(node.targets, node.value.elts):
                target_id = self.visit(target)
                value_id = self.visit(value)
                self.add_edge(value_id, target_id, '=')
        else:
            value_id = self.visit(node.value)
            for target in node.targets:
                target_id = self.visit(target)
                self.add_edge(value_id, target_id, '=')
    
    def visit_UnaryOp(self, node):
        operand_id = self.visit(node.operand)
        result_id = self.add_node(self.new_temp(), 'temp')
        op_type = type(node.op).__name__
        self.add_edge(operand_id, result_id, op_type)
        return result_id

    def visit_Compare(self, node):
        left_id = self.visit(node.left)
        result_id = self.add_node(self.new_temp(), 'temp')
        for operator, comparator in zip(node.ops, node.comparators):
            right_id = self.visit(comparator)
            op_type = type(operator).__name__
            self.add_edge(left_id, result_id, op_type)
            self.add_edge(right_id, result_id, op_type)
        return result_id

    def visit_BoolOp(self, node):
        result_id = self.add_node(self.new_temp(), 'temp')
        op_type = type(node.op).__name__
        for value in node.values:
            value_id = self.visit(value)
            self.add_edge(value_id, result_id, op_type)
        return result_id
    
    def visit_FunctionDef(self, node):
        self.function_defs[node.name] = [arg.arg for arg in node.args.args]
        self.generic_visit(node)
        

    def visit_Call(self, node):
        func_id = self.visit(node.func)
        result_id = self.add_node(self.new_temp(), 'temp')  
    
        if isinstance(node.func, ast.Name) and node.func.id in self.function_defs:
            formal_args = self.function_defs[node.func.id]
            for arg, formal_arg in zip(node.args, formal_args):
                arg_id = self.visit(arg)
                formal_arg_id = self.add_node(formal_arg, 'Name')
                self.add_edge(arg_id, formal_arg_id, 'Argument')
    
        return result_id

    def visit_Return(self, node):
        if node.value:
            value_id = self.visit(node.value)  
            return_id = self.add_node(self.new_temp(), 'temp')  
            self.add_edge(value_id, return_id, 'ReturnTo')
            return return_id
        return None

def build_dfg_from_code(code):
    try:
        parsed_ast = ast.parse(code)
        dfg_builder = DataFlowGraphBuilder()
        dfg_builder.visit(parsed_ast)

        node_list = [{'ID': node.id, 'name': node.name, 'kind': node.kind} 
                     for node_id, node in dfg_builder.nodes.items()]
        edge_list = [{'between': edge.between, 'edgeType': edge.edgeType} 
                     for edge in dfg_builder.edges]

        return node_list, edge_list
    except SyntaxError:
        print("Syntax error encountered. Skipping this code segment.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--datapath", default="./data/humaneval_python", help="data path")
    parser.add_argument("--output", default="./data/humaneval_graphs/dfg", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)   
    
    code_files = os.listdir(args.datapath)
    for code_f in tqdm(code_files):
        with open(os.path.join(args.datapath, code_f), 'r') as f:
            code = f.read()
            cfg = build_dfg_from_code(code)
            if cfg:
                with open(os.path.join(args.output, code_f[:-3]+'.pkl'), 'wb') as f:
                    pkl.dump(cfg, f)
        