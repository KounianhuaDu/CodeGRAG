import json

a = []
for chunk_interval in ["0:3500", "3500:7000", "7000:-1"]:
    cur_path = f"/home/jzchen/ML/AfterCodegrag/data/train/CodeContest/c++_{chunk_interval}_gemma.json"
    a += json.load(open(cur_path, "r"))
raw = json.load(
    open("/home/jzchen/ML/AfterCodegrag/data/train/CodeContest/c++.json", "r")
)
a = json.load(
    open("/home/jzchen/ML/AfterCodegrag/data/train/appsnew/python_Gemma_test.json", "r")
)
# # CodeForce
# a = json.load(open("/home/jzchen/ML/Code/data/train/CodeForce/c++_0:-1.json", "r"))
# print(len(a))
# assert 0
# # raw = json.load(open("/home/jzchen/ML/Code/data/train/CodeContest/c++.json", "r"))
# problem_file = (
#     "/home/jzchen/ML/Code/data/code4bench_filtered_A.jsonl.gz"  # 这里是测试数据的路径
# )
# from collections import defaultdict
# import gzip

# problems = defaultdict(dict)
# with gzip.open(problem_file, "rb") as f:
#     for line in f:
#         line = eval(line)  # 执行一个表达式，并返回表达式的值
#         problems[line["task_id"]] = (
#             line  # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
#         )

# print(raw[0])
fail_num = 0


for i in range(len(a)):
    if "Sure, here is the simplified problem:\n" in a[i]:
        idx = a[i].find("Sure, here is the simplified problem:\n")
        a[i] = a[i][idx + len("Sure, here is the simplified problem:\n") :]
    if "Sure, here is a simplified version of the problem:\n" in a[i]:
        idx = a[i].find("Sure, here is a simplified version of the problem:\n")
        a[i] = a[i][idx + len("Sure, here is a simplified version of the problem:\n") :]
    if "**Problem:**\n" in a[i]:
        idx = a[i].find("**Problem:**\n")
        a[i] = a[i][idx + len("**Problem:**\n") :]
    if "**Problem:**" in a[i]:
        idx = a[i].find("**Problem:**")
        a[i] = a[i][idx + len("**Problem:**") :]
    if "Here is the simplified problem:\n" in a[i]:
        idx = a[i].find("Here is the simplified problem:\n")
        a[i] = a[i][idx + len("Here is the simplified problem:\n") :]
    if "**Simplified Problem:**\n" in a[i]:
        idx = a[i].find("**Simplified Problem:**\n")
        a[i] = a[i][idx + len("**Simplified Problem:**\n") :]
    if a[i].startswith(" ") or a[i].startswith("\n"):
        a[i] = a[i][1:]
#     raw[i]["query"] = a[i]
# print(raw[0])
# json.dump(
#     raw,
#     open(
#         "/home/jzchen/ML/AfterCodegrag/data/train/CodeContest/c++extracted_Gemma.json",
#         "w",
#     ),
#     indent=4,
# )
print(a[0])
json.dump(
    a,
    open(
        "/home/jzchen/ML/AfterCodegrag/data/train/appsnew/python_Gemma_test.json", "w"
    ),
    indent=4,
)
print(fail_num)
