from openai import OpenAI

from datetime import datetime

client = OpenAI(
    # deepseek
    # api_key="sk-a0fef91da91143168555f2bb4be6a609", base_url="https://api.deepseek.com",
    
    # -----阿里云-----
    api_key="sk-bc4339fe8e8e48f6b57fab86b5d70afa", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

    # -----硅基流动----
    # api_key="sk-cvfuyotekxnioknxiplxfmsyxxjdbcfrhzdjkwecnybxgynd", base_url="https://api.siliconflow.cn/v1"
)

async def get_response(messages: list,
                       model= 'qwq-plus',#'qwen2.5-72b-instruct',#"deepseek-ai/DeepSeek-V3" , #"deepseek-chat",
                       strem=True,
                       # tools=tools
                       ):
    completion = client.chat.completions.create(
        temperature = 0.5,
        model = model,
        messages = messages,
        stream = strem,
        # stream_options = {"include_usage": True},
        # tools=tools,
        # extra_body={
        #     "enable_search": True  # 是否开启搜索功能，默认为False
        # }
    )
    return completion #.model_dump()

