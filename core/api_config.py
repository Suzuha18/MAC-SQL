import os
import openai

# ============================================================
#  多 API Provider 支持: deepseek / qwen / openai
#  通过环境变量 API_PROVIDER 切换，也可在 run.py 中通过命令行参数覆盖
# ============================================================

# 支持的 provider 默认配置
PROVIDER_CONFIGS = {
    "deepseek": {
        "api_base": "https://api.deepseek.com",
        "default_model": "deepseek-chat",        # 也可用 deepseek-reasoner
    },
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",             # 可选 qwen-turbo / qwen-plus / qwen-max / qwen-long
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4",
    },
}

# ---- 读取环境变量 ----
API_PROVIDER = os.getenv("API_PROVIDER", "deepseek").lower()  # deepseek / qwen / openai

# 允许用户通过 OPENAI_API_KEY 统一传 key（各家 key 都能用这个变量）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 如果显式设了 OPENAI_API_BASE 就用它，否则根据 provider 自动选
_provider_cfg = PROVIDER_CONFIGS.get(API_PROVIDER, PROVIDER_CONFIGS["deepseek"])
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", _provider_cfg["api_base"])

# ---- 初始化 openai 模块全局配置 ----
openai.api_type = "open_ai"
openai.api_base = OPENAI_API_BASE
openai.api_key = OPENAI_API_KEY
openai.api_version = None

# ---- 模型名称（可被 run.py --model_name 覆盖）----
MODEL_NAME = os.getenv("MODEL_NAME", _provider_cfg["default_model"])


def apply_provider_config(provider: str = None, model_name: str = None, api_key: str = None):
    """
    在运行时动态切换 provider / model_name / api_key。
    由 run.py 的命令行参数调用。
    """
    global API_PROVIDER, OPENAI_API_BASE, OPENAI_API_KEY, MODEL_NAME

    if provider:
        provider = provider.lower()
        API_PROVIDER = provider
        cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["deepseek"])
        OPENAI_API_BASE = cfg["api_base"]
        openai.api_base = OPENAI_API_BASE
        if model_name is None:
            MODEL_NAME = cfg["default_model"]

    if model_name:
        MODEL_NAME = model_name

    if api_key:
        OPENAI_API_KEY = api_key
        openai.api_key = OPENAI_API_KEY

    print(f"[api_config] provider={API_PROVIDER}, model={MODEL_NAME}, api_base={OPENAI_API_BASE}")