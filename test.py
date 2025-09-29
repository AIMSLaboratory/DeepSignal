import openai
from openai import OpenAI

# 配置本地服务端参数
client = OpenAI(
    base_url="http://10.147.18.148:1234/v1",  # LM Studio 默认端口
    api_key="lm-studio"  # 无需真实密钥，但必须填写（LM Studio不验证）
)

def get_local_response(messages, model_name="qwen3-8b", stream=False):
    """获取本地模型响应（兼容OpenAI格式）"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=stream
        )
        return response
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Always answer in rhymes. Today is Thursday"},
        {"role": "user", "content": "What day is it today?"}
    ]
    
    # 非流式调用
    response = get_local_response(messages)
    print(response)
    print(response.choices[0].message.content)

    # 流式调用
    stream_response = get_local_response(messages, stream=True)
    for chunk in stream_response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)