import os
import sys
sys.path.append("..")
import argparse
from utils.evaluation import evaluate_functional_correctness

def evaluation(args):
    result = evaluate_functional_correctness(
        input_file=args.output,
        tmp_dir=args.res,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=args.lang
    )
    print(args.lang, result)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--problem", default="/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz", help="data path")
    parser.add_argument("--output", default="/home/knhdu/output/FinalVersion/samples_with_graph_1.jsonl", help="output path")
    parser.add_argument("--res", default="/home/knhdu/1132/", help="output path")
    parser.add_argument("--lang", default="cpp", help="output path")
    
    args = parser.parse_args()

    problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    
    evaluation(args)
