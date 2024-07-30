import json
import fire
import gzip
import os
import pandas as pd
from pprint import pprint
import pickle as pkl


def prompt_filter(datapoint):
    example_idx = datapoint.find("Examples")
    datapoint = datapoint[:example_idx]
    return datapoint


def solution_filter_cpp(datapoint):
    languages = datapoint["language"]
    solutions = datapoint["solution"][languages == 2]
    if len(solutions):
        return min(solutions, key=lambda x: len(x))
    else:
        return None


def solution_filter_python(datapoint):
    languages = datapoint["language"]
    solutions = datapoint["solution"][languages == 3]
    if len(solutions):
        return min(solutions, key=lambda x: len(x))
    else:
        return None


def main(language="c++", data_path="../data/train/", dataset="CodeContest"):
    full_path = os.path.join(data_path, dataset)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    print(f'Results will be saved at {full_path}')
    if dataset == "CodeContest":
        problem_file = "../raw_data/CodeContest"
    else:
        raise NotImplementedError
    fp = os.path.join(data_path, f"{dataset}", f"{language}.json")
    problems = []
    
    print("Loading code")
    with open(os.path.join(f"../raw_data/{dataset}/tr/", "codes.pkl"), "rb") as f:
        codes = pkl.load(f)

    problem_id = 0
    raw_file_path = f"../raw_data/{dataset}/raw_files_graph/"
    for filename in os.listdir(problem_file):
        if filename.startswith("train"):
            df = pd.read_parquet(os.path.join(problem_file, filename))
            df["description"] = df["description"].apply(lambda x: prompt_filter(x))
            if language == "c++":
                df["solutions"] = df["solutions"].apply(
                    lambda x: solution_filter_cpp(x)
                )
            elif language == "python":
                df["solutions"] = df["solutions"].apply(
                    lambda x: solution_filter_python(x)
                )
            elif language == "java":
                pass
            else:
                raise NotImplementedError
            for idx, datapoint in df.iterrows():
                if not datapoint["solutions"]:
                    continue
                cur_datapoint = {}
                cur_datapoint["input"] = datapoint["description"]
                cur_datapoint["output"] = datapoint["solutions"]
                if cur_datapoint["output"] in codes:  # Can be in train dataset
                    # Write as cpp files for graph extraction
                    with open(
                        os.path.join(raw_file_path, f"{problem_id}.cpp"), "w"
                    ) as f:
                        f.write(datapoint["solutions"])
                    cur_datapoint["index"] = problem_id
                    problem_id += 1
                    problems.append(cur_datapoint)
    print(f"Dataset lenth {len(problems)}")
    json.dump(problems, open(fp, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(main)
