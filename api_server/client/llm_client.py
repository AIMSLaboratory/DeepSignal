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

# Traffic-R1 模型相关导入
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers/torch 未安装，无法使用 Traffic-R1 模型")

class LLMClient:
    def __init__(self, model_type: str = model_type):
        """
        初始化LLM客户端
        Args:
            model_type: 模型类型，可选值：openai, anthropic, siliconflow, traffic_r1, local_gguf, lm-studio
        """
        self.model_type = model_type
        self.client = None
        self.model = None  # 用于 transformers 模型
        self.tokenizer = None  # 用于 transformers tokenizer
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
        elif self.model_type == "traffic_r1":
            # Traffic-R1 本地模型初始化
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("❌ 需要安装 transformers 和 torch 才能使用 Traffic-R1 模型")
            
            model_path = os.getenv('TRAFFIC_R1_MODEL_PATH', './models/Traffic-R1')
            print(f"🚀 正在加载 Traffic-R1 模型: {model_path}")
            
            # 检测设备
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                print("✅ 使用 MPS (Apple Silicon) 加速")
            elif torch.cuda.is_available():
                device = "cuda"
                print("✅ 使用 CUDA 加速")
            else:
                device = "cpu"
                print("⚠️ 使用 CPU 运行（较慢）")
            
            # 加载模型和tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                device_map=device if device != "mps" else None,  # MPS不支持device_map
                trust_remote_code=True
            )
            
            # 如果是MPS，手动移动模型
            if device == "mps":
                self.model = self.model.to(device)
            
            self.model.eval()  # 设置为评估模式
            print(f"✅ Traffic-R1 模型加载成功！设备: {device}")
            
        elif self.model_type == "local_gguf":
            # GGUF格式模型初始化
            from llama_cpp import Llama
            self.client = Llama(
                model_path=os.getenv('LOCAL_GGUF_MODEL_PATH', './models/qwen3-8b.Q4_K_M.gguf'),
                n_ctx=4096,
                n_gpu_layers=1,  # Metal加速
                verbose=False
            )
        elif self.model_type == "local" or self.model_type == "lm-studio":
            # 本地LM-Studio模式
            self.client = OpenAI(
                api_key=os.getenv('LM_STUDIO_API_KEY', 'lm-studio'),
                base_url=os.getenv('LM_STUDIO_BASE_URL', 'http://127.0.0.1:1234/v1')
            )
            print(f"✅ 本地LM-Studio已连接: {self.client.base_url}")
        else:
            raise ValueError(f"❌ 不支持的模型类型: {self.model_type}")
        
        

    def chat(self, messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 100,
             model: str = None) -> Dict[str, Any]:
        """
        同步方法：获取模型回复（不使用流式输出，用于信号控制）
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            model: 模型名称
        
        Returns:
            包含content的字典
        """
        model = model or os.getenv('MODEL_NAME', 'qwen-plus')
        
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    temperature=temperature,
                    model=model,
                    messages=messages,
                    stream=False,
                    max_tokens=max_tokens
                )
                return {"content": response.choices[0].message.content}
                
            elif self.model_type == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=temperature
                )
                return {"content": response.content[0].text}
                
            elif self.model_type == "siliconflow":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {"content": response.choices[0].message.content}
                
            elif self.model_type == "traffic_r1":
                # Traffic-R1 模型推理
                # 构建输入文本
                text = self._format_messages_for_traffic_r1(messages)
                
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt")
                
                # 移动到正确的设备
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码
                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 去除输入部分，只保留生成的内容
                if text in response_text:
                    response_text = response_text[len(text):].strip()
                
                return {"content": response_text}
                
            elif self.model_type == "local" or self.model_type == "lm-studio":
                response = self.client.chat.completions.create(
                    model=model or "local-model",
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {"content": response.choices[0].message.content}
            else:
                raise ValueError(f"❌ 不支持的模型类型: {self.model_type}")
                
        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")
            raise
    
    def _format_messages_for_traffic_r1(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化消息为 Traffic-R1 模型的输入格式
        
        Args:
            messages: 消息列表
        
        Returns:
            格式化后的文本
        """
        # Qwen2 格式的对话模板
        formatted_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # 添加最后的 assistant 标记，等待模型生成
        formatted_text += "<|im_start|>assistant\n"
        
        return formatted_text

    async def get_chat_response(self, 
                           messages: List[Dict[str, str]], 
                           model: str = os.getenv('MODEL_NAME'),
                           stream: bool = True,
                           tools: Optional[List[Dict[str, Any]]] = tools
                           ) -> Any:
        """
        异步方法：获取模型回复（流式输出，用于对话）
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
        elif self.model_type == "local" or self.model_type == "lm-studio":
            return self.client.chat.completions.create(
                model=model or "local-model",
                messages=messages,
                stream=stream,
                tools=tools
            )
        else:
            raise ValueError(f"❌ 不支持的模型类型: {self.model_type}")
        

if __name__ == "__main__":
    mcp_client = get_mcp_client()
    print(mcp_client.sync_list_tools())