# import os
# # set your OPENAI_API_BASE, OPENAI_API_KEY here!
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "your_own_api_base")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_own_api_key")
#
# import openai
# openai.api_type = "azure"
# openai.api_base = OPENAI_API_BASE
# # set your own api_version
# openai.api_version = "2024-07-01-preview"
# openai.api_key = OPENAI_API_KEY
#
# MODEL_NAME = 'gpt-4-1106-preview' # 128k 版本
# # MODEL_NAME = 'CodeLlama-7b-hf'
# # MODEL_NAME = 'gpt-4-32k' # 0613版本
# # MODEL_NAME = 'gpt-4' # 0613版本
# # MODEL_NAME = 'gpt-35-turbo-16k' # 0613版本

import os

# 1. 设置 DeepSeek 的 Base URL 和 Key
# 建议直接写在这里，或者确保环境变量已正确设置
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-2a72101b2aa14cd99e7c7e8ad2bde60d")

import openai

# 2. 修改 API 类型为标准 open_ai (移除 azure)
openai.api_type = "open_ai"
openai.api_base = OPENAI_API_BASE
openai.api_key = OPENAI_API_KEY

# 3. DeepSeek 不需要 Azure 的 api_version，建议注释掉或设为 None
# openai.api_version = "2023-07-01-preview"
openai.api_version = None

# 4. 修改模型名称
MODEL_NAME = 'deepseek-chat'  # 或者 'deepseek-reasoner'
# MODEL_NAME = 'gpt-4-1106-preview'