import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import sys
import os
import asyncio

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api_server.client.mcp_client import get_mcp_client,get_mcp_client_async

load_dotenv()  # load environment variables from .env

# mcp_client = get_mcp_client_async()
tools = None
model_type = os.getenv('MODEL_TYPE', 'openai')

class LLMClient:
    def __init__(self, model_type: str = model_type):
        """
        初始化LLM客户端
        Args:
            model_type: 模型类型，可选值：openai, anthropic, siliconflow
        """
        self.model_type = model_type
        self.client = None
        self._init_client()
        self.mcp_client = None
        self.tools = []
    
    async def initialize(self):
        """异步初始化MCP客户端"""
        print("调用MCP连接")
        self.mcp_client = await get_mcp_client_async()
        
    def _init_client(self):
        """初始化对应的客户端"""
        if self.model_type == "openai":
            self.client = OpenAI(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif self.model_type == "anthropic":
            self.client = Anthropic()
        elif self.model_type == "siliconflow":
            self.client = OpenAI(
                api_key=os.getenv('SILICONFLOW_API_KEY', ''),
                base_url=os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
            )
        elif self.model_type == "local_gguf":
            # GGUF格式模型初始化
            from llama_cpp import Llama
            self.client = Llama(
                model_path=os.getenv('LOCAL_GGUF_MODEL_PATH', './models/qwen3-8b.Q4_K_M.gguf'),
                n_ctx=4096,
                n_gpu_layers=1,  # Metal加速
                verbose=False
            )
        elif self.model_type == model_type:
            self.client = OpenAI(
                api_key="lm-studio",
                base_url="http://10.147.18.148:1234/v1"
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        

    async def get_chat_response(self, 
                           messages: List[Dict[str, str]], 
                           model: str = os.getenv('MODEL_NAME'),
                           stream: bool = True,
                           tools: Optional[List[Dict[str, Any]]] = tools
                           ) -> Any:
        """
        获取模型回复
        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否使用流式输出
            tools: 可用工具列表
        Returns:
            模型回复
        """
        await self.initialize()
        tools = await self.mcp_client.get_list_tools()
        if self.model_type == "openai":
            return self.client.chat.completions.create(
                temperature=0.5,
                model=model,
                messages=messages,
                stream=stream,
                tools=tools
            )
        elif self.model_type == "anthropic":
            return self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages
            )
        elif self.model_type == "siliconflow":
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                tools=tools
            )
        elif self.model_type == model_type:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                tools=tools
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}") 
        

if __name__ == "__main__":
    mcp_client = get_mcp_client()
    print(mcp_client.sync_list_tools())