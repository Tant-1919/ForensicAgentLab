"""Local model settings. Credentials stay in the original file or environment."""
import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ENV_KEYS = ("AGENT_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL")
DASHSCOPE_HOSTS = {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
                   "cn-hongkong.dashscope.aliyuncs.com"}

def live_model_settings():
    """Choose a complete manual setup, or reuse an explicitly selected Java YAML."""
    explicit = [os.getenv(key, "").strip() for key in ENV_KEYS]
    if explicit[0] or explicit[1]:
        if not all(explicit):
            raise RuntimeError("请同时设置 AGENT_API_KEY、AGENT_BASE_URL 和 AGENT_MODEL")
        return dict(zip(("api_key", "base_url", "model"), explicit, strict=True))

    config = os.getenv("AGENT_JAVA_CONFIG", "").strip()
    if not config:
        raise RuntimeError("请设置 AGENT_JAVA_CONFIG 或配置完整的模型环境变量")
    try:
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8-sig"))
        aliyun = data["aliyun"]
        key = aliyun["api-key"]
        model = explicit[2] or aliyun["text"]["model"]
        url = urlsplit(aliyun["text"]["url"])
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        # A YAML parser error can include the original source line. Do not echo it.
        raise RuntimeError("原 Java YAML 读取失败，或缺少 aliyun 文本模型设置") from None

    host = url.hostname or ""
    # Translate the native DashScope path to the compatible path on the same
    # trusted host; never forward a key to a URL copied from arbitrary YAML.
    allowed = host in DASHSCOPE_HOSTS or host.endswith(".maas.aliyuncs.com")
    if url.scheme != "https" or not allowed or url.netloc != host:
        raise RuntimeError("原 Java 文本模型地址不属于受支持的 DashScope HTTPS 域名")
    if not all(isinstance(x, str) and x.strip() and not x.strip().startswith("${") for x in (key, model)):
        raise RuntimeError("原 Java YAML 未提供可直接复用的密钥或模型名，请配置模型环境变量")
    return {"api_key": key.strip(), "model": model.strip(),
            "base_url": f"https://{host}/compatible-mode/v1"}
