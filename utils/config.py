import platform
import logging
import os
import openai

#APEX
API_KEY = "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"
#Huawei
#API_KEY = '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'
openai.api_key = API_KEY

#logging.basicConfig(format='%(levelname)s %(asctime)s %(process)d %(message)s',level=logging.INFO)
system = platform.system()
os.environ["http_proxy"] = "http://127.0.0.1:8888"
os.environ["https_proxy"] = "http://127.0.0.1:8888"
os.environ["all_proxy"] = "socks5://127.0.0.1:8889"
os.environ["OPENAI_API_KEY"] = API_KEY

