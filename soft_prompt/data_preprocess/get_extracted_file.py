import json

a = []
for chunk_interval in [
    "0:1000",
    "1000:2000",
    "2000:3000",
    "3000:4000",
    "4000:5000",
    "5000:6000",
    "6000:7000",
    "7000:8000",
    "8000:-1",
]:
    cur_path = f"/home/jzchen/ML/Code/data/train/appsnew/python_{chunk_interval}.json"
    a += json.load(open(cur_path, "r"))
raw = json.load(open("/home/jzchen/ML/Code/data/train/appsnew/python.json", "r"))
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

# CodeForce
for i in range(len(a)):
    if "[INST]" in a[i] or isinstance(a[i], list):
        fail_num += 1
        raw[i]["query"] = raw[i]["input"]
        continue
    elif "# Simplified Problem:\n\n" in a[i]:
        a[i] = a[i][len("# Simplified Problem:\n\n") :]
    if "# Simplified Problem\n\n" in a[i]:
        a[i] = a[i][len("# Simplified Problem\n\n") :]
    if "# The problem is:\n\n" in a[i]:
        a[i] = a[i][len("# The problem is:\n\n") :]
    if a[i].startswith("#"):
        a[i] = a[i][1:]
    if a[i].startswith(" "):
        a[i] = a[i][1:]
    raw[i]["query"] = a[i]
print(raw[0])
json.dump(
    raw,
    open("/home/jzchen/ML/Code/data/train/appsnew/pythonextracted.json", "w"),
    indent=4,
)
print(fail_num)
