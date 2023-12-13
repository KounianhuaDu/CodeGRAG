import platform
import logging
import os
import openai

API_KEY = "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"
#API_KEY = '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'
openai.api_key = API_KEY
logging.basicConfig(format='%(levelname)s %(asctime)s %(process)d %(message)s',level=logging.INFO)

system = platform.system()
if system == "Linux":
    os.environ["http_proxy"] = "http://127.0.0.1:8888"
    os.environ["https_proxy"] = "http://127.0.0.1:8888"
    os.environ["all_proxy"] = "socks5://127.0.0.1:8889"

qa_file_path = {
        "wiki-C": '/home/share/CodeMateData/QAData/crawler/dd_wiki-C-492.jsonl',
        "wiki-Alg":'/home/share/CodeMateData/QAData/crawler/dd_wiki-Alg-489.jsonl',
        "stackoverflow-DS":'/home/share/CodeMateData/QAData/crawler/dd_stackoverflow-DS-33124.jsonl',
        "stackoverflow-Alg":'/home/share/CodeMateData/QAData/crawler/dd_stackoverflow-Alg-11561.jsonl',
        "stackoverflow-C":'/home/share/CodeMateData/QAData/crawler/dd_stackoverflow-C-38405.jsonl',
        "CSDN-DS":'/home/share/CodeMateData/QAData/crawler/dd_filteredCSDN-DS-15201.jsonl',
        "CSDN-C":'/home/share/CodeMateData/QAData/crawler/dd_filteredCSDN-C-10746.jsonl',
        "CSDN-Alg":'/home/share/CodeMateData/QAData/crawler/dd_filteredCSDN-Alg-5718.jsonl',
        "leetcode":'/home/share/CodeMateData/QAData/crawler/leetcode_2442.jsonl',
        "high-train":'/home/share/CodeMateData/QAData/crawler/trainingData/highQualityTrain.jsonl',
        "high-test":'/home/share/CodeMateData/QAData/crawler/trainingData/highQualityTest.jsonl',
        "BELLE2":'/home/share/CodeMateData/QAData/dataset_filtered/Belle2_Positive-83016.jsonl',
        "BELLE":'/home/share/CodeMateData/QAData/dataset_filtered/Belle_Positive_1809.jsonl',
        "moss":'/home/share/CodeMateData/QAData/dataset_filtered/filtered-moss-1187390.jsonl',
        "total":'/home/knhdu/Code/data/retrieval_models/QA_total.json',
        "wiki":'/home/knhdu/Code/data/retrieval_models/wiki.json',
        "stackoverflow":'/home/knhdu/Code/data/retrieval_models/stackoverflow.json',
        "textbook":'/home/wmzhang/codemate/newtest/textbook/textbook.json',
        "filtered":'/home/knhdu/Code/data/retrieval_models/filtered.json',
        "codes": '/home/wmzhang/codemate/codeT5+/output.json',
        "codes_total": "/home/rrt/codemate/CodeTest/codes/codes_total.json" 
        }

emb_path = {
        "wiki-C": '/home/wmzhang/codemate/newtest/retrieval_models/wiki-C.npy',
        "wiki-Alg":'/home/wmzhang/codemate/newtest/retrieval_models/wiki-Alg.npy',
        "stackoverflow-DS":'/home/wmzhang/codemate/newtest/retrieval_models/stackoverflow-DS.npy',
        "stackoverflow-Alg":'/home/wmzhang/codemate/newtest/retrieval_models/stackoverflow-Alg.npy',
        "stackoverflow-C":'/home/wmzhang/codemate/newtest/retrieval_models/stackoverflow-C.npy',
        "CSDN-DS":'/home/wmzhang/codemate/newtest/retrieval_models/CSDN-DS.npy',
        "CSDN-C":'/home/wmzhang/codemate/newtest/retrieval_models/CSDN-C.npy',
        "CSDN-Alg":'/home/wmzhang/codemate/newtest/retrieval_models/CSDN-Alg.npy',
        "leetcode":'/home/wmzhang/codemate/newtest/retrieval_models/leetcode.npy',
        "high-train":'/home/wmzhang/codemate/newtest/retrieval_models/high-train.npy',
        "high-test":'/home/wmzhang/codemate/newtest/retrieval_models/high-test.npy',
        "BELLE2":'/home/wmzhang/codemate/newtest/retrieval_models/BELLE2.npy',
        "BELLE":'/home/wmzhang/codemate/newtest/retrieval_models/BELLE.npy',
        "moss":'/home/wmzhang/codemate/newtest/retrieval_models/moss.npy',
        "total":'/home/knhdu/Code/data/retrieval_models/total.npy',
        "wiki":'/home/knhdu/Code/data/retrieval_models/wiki.npy',
        "stackoverflow":'/home/knhdu/Code/data/retrieval_models/stackoverflow.npy',
        "textbook":'/home/wmzhang/codemate/newtest/textbook/textbook.npy',
        "filtered":'/home/knhdu/Code/data/retrieval_models/filtered.npy',
        "codes": '/home/wmzhang/codemate/codeT5+/code.npy',
        "codes_total": "/home/rrt/codemate/CodeTest/codes/codes_total.npy"
                }

knowledge_prompt_prefix = """根据问题描述输出这道题的相关知识点，不需要代码。
问题描述：%s
输入：%s
输出：%s
测试示例：%s
函数声明：%s
"""

chinese_prompt_prefix_v0 = """我会给你一个需要用C++编程解决的问题和一个C++的函数声明，你的任务是按这个函数声明生成对应的C++函数去解决这个问题，你只需要返回C++函数,不要返回额外的说明文字。
问题：%s
函数声明：%s 
"""

english_prompt_prefix_v0 = """I will provide you with a problem that needs to be solved using C++ programming, along with a C++ function declaration. Your task is to generate the corresponding C++ function based on this function declaration to solve the given problem. You only need to return the C++ function, without any additional explanatory text.
Problem: %s
Function Declaration: %s 
"""

chinese_prompt_prefix_v1 = """我会给你一个用于解决某个问题但缺少了函数定义的C++代码，这个缺失函数定义的函数声明，这个C++代码解决的问题，
你的任务是用这个函数声明生成函数定义去填补所给的C++代码缺少的部分，你只需要返回C++代码的函数定义,不要返回额外的说明文字。
问题：%s
代码：%s
函数声明：%s
"""

english_prompt_prefix_v1 = """I will provide you with a piece of C++ code that is meant to solve a certain problem but lacks a function definition. This missing function definition is actually a function declaration. The C++ code addresses a specific problem. Your task is to take this function declaration and create the corresponding function definition to fill in the missing part of the given C++ code. You only need to return the function definition for the C++ code, without providing additional explanatory text.
Problem: %s
Code: %s
Function Declaration: %s
"""

chinese_prompt_prefix_v2 = """我会给你一个需要用C++编程解决的问题和一个C++的函数声明，你的任务是按这个函数声明生成对应的C++函数去解决这个问题，并满足时间限制和空间限制。
你只需要返回C++函数,不要返回额外的说明文字。
问题：%s
函数声明：%s 
时间限制：%s ms
空间限制：%s MB
"""

english_prompt_prefix_v2 = """I will provide you with a problem that needs to be solved using C++ programming, along with a function declaration in C++. Your task is to create the corresponding C++ function based on this declaration to solve the problem, while adhering to the time and space constraints. You only need to return the C++ function itself, without providing additional explanatory text.
Problem: %s
Function Declaration: %s
Time Limit: %s ms
Space Limit: %s MB
"""

chinese_prompt_prefix_v3 = """根据问题描述编写一个符合函数声明的C++的函数，编译正确且通过测试示例。你只需要返回C++函数，不需要额外的说明文字。
问题描述：%s
输入：%s
输出：%s
测试示例：%s
函数声明：%s
"""

chinese_prompt_prefix_v4 = """根据问题描述编写一个符合函数声明的C++的函数，编译正确且通过测试示例。你只需要返回C++函数，不需要额外的说明文字。
问题描述：%s
输入：%s
输出：%s
测试示例：%s
函数声明：%s
提示：%s
"""

chinese_prompt_prefix_v6 = """根据问题描述编写一个符合函数声明的C++的函数，编译正确且通过测试示例。你只需要返回C++函数，不需要额外的说明文字。
问题描述：%s
输入：%s
输出：%s
测试示例：%s
函数声明：%s
提示：%s
"""

