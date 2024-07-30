import json
import gzip
import os
from collections import defaultdict
from datasets import load_dataset
from codegeex.data.data_utils import write_jsonl
import sys
import re
import warnings

sys.path.append("../")
from support.Search_with_CodeT5 import construct_faiss_index, search_with_faiss
import numpy as np
from tqdm import tqdm


class DataDealer:
    def __init__(
        self,
        dataset,
        split,
        extracted=True,
        softprompt=True,
        cutofflen=3600,
        extracted_token="extracted_Gemma",
        retrieval_method="NL-CL",
    ):
        self.extracted = extracted
        self.split = split
        self.cutofflen = cutofflen
        self.softprompt = softprompt
        self.dataset = dataset
        self.extracted_token = extracted_token if self.extracted else ""
        if dataset == "appsnew":
            if split == "train":
                self.data_path = f"/home/jzchen/ML/AfterCodegrag/data/train/appsnew/python{self.extracted_token}.json"
            else:  # Test dataset apps
                self.data_path = (
                    "/home/jzchen/ML/AfterCodegrag/data/apps_sampled.jsonl.gz"
                )
        elif dataset == "CodeContest" and split == "train":
            self.data_path = f"/home/jzchen/ML/AfterCodegrag/data/train/CodeContest/c++{self.extracted_token}.json"
        else:  # CodeForce
            self.data_path = (
                "/home/jzchen/ML/AfterCodegrag/data/code4bench_filtered_A.jsonl.gz"
            )

        self.problems = None
        if dataset == "CodeContest" or dataset == "CodeForce":
            self.language = "c++"
        else:
            self.language = "python"
        self.load_data()
        if self.softprompt:
            if retrieval_method == "NL-CL":
                self.code_emb = np.load(
                    f"/home/jzchen/ML/AfterCodegrag/raw_data/{dataset}/dgl/codes_emb.npy"
                )
            else:
                assert self.language == "c++"
                self.code_emb = np.load(
                    "/home/jzchen/ML/Code/data/train/CodeContest/emb/nls.npy"
                )
            # Build faiss index
            self.index, self.pca = construct_faiss_index(self.code_emb)

            # Index table of Graph embedding
            self.graph_emb_list = np.load(
                f"/home/jzchen/ML/AfterCodegrag/raw_data/{dataset}/dgl/graphs_emb.npy"
            )

    def get_datapath(self):
        return self.data_path

    def load_data(self):
        if self.data_path.endswith(".json"):
            self.problems = json.load(open(self.data_path, "r"))
        elif self.data_path.endswith(".gz"):
            self.problems = defaultdict(dict)
            with gzip.open(self.data_path, "rb") as f:
                for line in f:
                    line = eval(line)  # 执行一个表达式，并返回表达式的值
                    self.problems[line["task_id"]] = (
                        line  # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
                    )

    def give_train_dataset_for_transformers(self):
        assert self.split == "train", "ERROR: not in training mode"
        DATA_PATH = {"train": self.data_path}
        dataset = load_dataset("json", data_files=DATA_PATH)
        print("Data loaded")
        return dataset

    def get_prompt_description(self, chunk_interval):
        """
        Called in extraction process, return a List object for extraction
        """
        start_idx, end_idx = int(chunk_interval.split(":")[0]), int(
            chunk_interval.split(":")[1]
        )
        assert self.problems, "Must call load data function first"
        if isinstance(self.problems, list):
            if end_idx != -1:
                descriptions = [question["input"] for question in self.problems][
                    start_idx:end_idx
                ]
            else:
                descriptions = [question["input"] for question in self.problems][
                    start_idx:
                ]
        elif isinstance(self.problems, defaultdict):
            if end_idx != -1:
                descriptions = [
                    question["prompt"] for question in self.problems.values()
                ][start_idx:end_idx]
            else:
                descriptions = [
                    question["prompt"] for question in self.problems.values()
                ][start_idx:]
        else:
            raise NotImplementedError
        assert isinstance(descriptions, list)
        return descriptions

    def extraction_fileter(self, description, answer):
        """
        Input: raw description:List, answer by LLM: List
        return: Well formated answer list
        """
        assert len(description) == len(answer)
        fail_num = 0
        for i in range(len(answer)):
            if "[INST]" in answer[i] or isinstance(answer[i], list):
                fail_num += 1
                answer[i] = description[i]
                continue
            if "# Simplified Problem:\n\n" in answer[i]:
                answer[i] = answer[i][len("# Simplified Problem:\n\n") :]
            if "# Simplified Problem\n\n" in answer[i]:
                answer[i] = answer[i][len("# Simplified Problem\n\n") :]
            if "# The problem is:\n\n" in answer[i]:
                answer[i] = answer[i][len("# The problem is:\n\n") :]
            if "Sure, here is the simplified problem:\n" in answer[i]:
                answer[i] = answer[i][len("Sure, here is the simplified problem:\n") :]
            if "**Problem:**\n" in answer[i]:
                answer[i] = answer[i][len("**Problem:**\n") :]
            if "**Simplified Problem:**\n" in answer[i]:
                answer[i] = answer[i][len("**Simplified Problem:**\n") :]
            if answer[i].startswith("#"):
                answer[i] = answer[i][1:]
            if answer[i].startswith(" "):
                answer[i] = answer[i][1:]
            answer[i] = answer[i].replace("\n\n", "\n")
        return answer

    def save_as_json(self, object, path):
        with open(
            path,
            "w",
        ) as f:
            json.dump(object, f)

    def iter_test_data(self):
        assert self.split == "test", "Should be in test mode"
        print(f"Total dataset length {len(self.problems)}")
        for task_id in tqdm(self.problems):
            problem = self.problems[task_id]
            yield task_id, problem

    def output_filter(self, output, model="Llama2"):
        if model == "Llama2" or model == "codellama":
            start_pos = output.find("[/INST]")
            if start_pos != -1:
                completion = output[start_pos + len("[/INST]") :].strip()
                failed = 0
                pattern = r"```(.*?)```"
                matches = re.findall(pattern, output, re.DOTALL)
                if matches:
                    completion = matches[0].strip()
                else:
                    pattern = re.compile(
                        # r"Answer:\s*(#include\s+<.*?>.*?return\s+0;\s*}\s*)", re.DOTALL
                        r"(#include\s+<.*?>.*?\s*}\s*)",
                        re.DOTALL,
                    )
                    match = pattern.search(output)
                    if match:
                        completion = match.group(1)
                        failed = 0

            else:
                print("****************Code extraction failed!*****************")
                print(output)
                failed = 1
                completion = output
        elif model == "Gemma":
            pattern = rf"```{self.language}(.*?)```"
            matches = re.findall(pattern, output, re.DOTALL)
            if matches:
                completion = matches[0].strip()
                failed = 0
            else:
                pattern = r"<start_of_turn>model(.*?)<end_of_turn>"
                matches = re.findall(pattern, output, re.DOTALL)
                if matches:
                    completion = matches[0].strip()
                    failed = 0

                else:
                    print("****************Code extraction failed!*****************")
                    print(output)
                    failed = 1
                    completion = output
            # raw_code = re.findall(f"(?is)model(.*)", output)
            # if raw_code:
            #     codes = raw_code[0].strip()
            #     codes = codes.split("\n")
            #     for i in range(len(codes)):
            #         if codes[i] != "model":
            #             completion = "\n".join(codes[i:])
            #             break
            #     failed = 0
            # else:
            #     pattern = rf"```{self.language}(.*?)```"
            #     matches = re.findall(pattern, output, re.DOTALL)
            #     if matches:
            #         completion = matches[0].strip()
            #         failed = 0
            #     else:
            #         pattern = r"<start_of_turn>model(.*?)<end_of_turn>"
            #         matches = re.findall(pattern, output, re.DOTALL)
            #         if matches:
            #             completion = matches[0].strip()
            #             failed = 0

            #         else:
            #             print(
            #                 "****************Code extraction failed!*****************"
            #             )
            #             print(output)
            #             failed = 1
            #             completion = output
        elif model == "starcoder":
            if self.language == "c++":
                pattern = re.compile(
                    # r"Answer:\s*(#include\s+<.*?>.*?return\s+0;\s*}\s*)", re.DOTALL
                    r"Answer:\s*(#include\s+<.*?>.*?\s*}\s*)",
                    re.DOTALL,
                )
                match = pattern.search(output)
                if match:
                    completion = match.group(1)
                    failed = 0
                else:
                    print("****************Code extraction failed!*****************")
                    print(output)
                    failed = 1
                    completion = output
        return failed, completion

    def formatted_return(self, completion, problem, task_id):
        ans = {}
        if "apps" in self.dataset:  # Apps Dataset
            ans[int(task_id)] = completion
        else:  # CodeForce Dataset
            ans = dict(
                task_id=task_id,
                generation=completion,
                prompt=problem["prompt"],
                example_testcases=problem["example_testcases"],
                testcases=problem["testcases"],
            )
        return ans

    def save_results(self, object, path):
        assert self.split == "test", "Must be called in test mode"
        if "apps" in self.dataset:
            if path.endswith("jsonl"):
                warnings.warn("apps dataset requires a .json file, renaming path")
                path = path.replace("jsonl", "json")
            samples = {}
            for ans in object:
                samples[list(ans.keys())[0]] = list(ans.values())[0]
            self.save_as_json(samples, path)
        else:
            write_jsonl(path, object)
        print("Results saved at", path)

    def get_extracted_test_list(self):
        assert self.split == "test", "Must be call in testing mode"
        if self.language == "python":
            return json.load(
                open(
                    "/home/jzchen/ML/AfterCodegrag/data/train/appsnew/python_Gemma_test.json",
                    "r",
                )
            )
        elif self.language == "c++":
            return json.load(
                open(
                    "/home/jzchen/ML/AfterCodegrag/data/train/CodeForce/c++_Gemma_test.json",
                    "r",
                )
            )
