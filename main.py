# from build.lib.traci import junction
# from click import prompt
from fastapi import FastAPI
from pydantic import BaseModel
from api_server.utils.run_command import run_command
import os
import asyncio
from loguru import logger
from sumo_llm.sumo_simulator import initialize_sumo, get_simulator,stop_simulation

from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from openai_client import get_response
import json
from fastapi.middleware.cors import CORSMiddleware
from api_server.client.llm_client import LLMClient
from api_server.client.mcp_client import get_mcp_client,get_mcp_client_async
from dotenv import load_dotenv


# 添加定时任务相关的变量
background_tasks = set()
load_dotenv()
os.path.abspath(__file__)
print(os.getcwd())

app = FastAPI()


origins = [
    "http://127.0.0.1:80", "http://127.0.0.1:9080", "http://127.0.0.1", "http://192.168.0.103:83", "192.168.1.13:81",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# os.chdir(r"E:/wq/2025/交通AI/sumo-llm/api-server")

class Item(BaseModel):
    name: str
    price: float

prompt = f"""作为交通信号配时专家，请根据数据进行信号与相位优化，可以根据相关数据自行推理判断优化或调用相关算法结果优化，路口描述：
该路口是成都市的"倪家桥路与领事馆路交叉口"。有4个相位，分别是：
0: "南北方向直行与右转",
    1: "南北方向左转",
    2: "东西方向直行与右转",
    3: "东西方向左转 
    直接给出优化结果即可。
    """

async def periodic_chat_task():
    while True:
        try:
            # 创建模拟的请求对象
            class MockRequest:
                async def json(self):
                    return {
                        "message": "优化当前相位"
                    }
                async def is_disconnected(self):
                    return False

            mock_request = MockRequest()
            # 准备消息列表
            system_message = [{'role': 'system', 'content': prompt}]
            junction_state_data = [['倪家桥路与领事馆路交叉口,信号灯ID:J54']]
            # 使用 json.dumps 确保生成有效的 JSON 字符串
            system_message.append({'role': 'system', 'content': json.dumps(junction_state_data, ensure_ascii=False)})
            messages = system_message + [{'role': 'user', 'content': '优化当前相位'}]
            
            # 正确调用 call_with_messages
            async for _ in call_with_messages(messages, mock_request):
                pass
        except Exception as e:
            logger.error(f"定时任务执行出错: {str(e)}")
        await asyncio.sleep(10)  # 每10秒执行一次

@app.on_event("startup")
async def startup_event():
    global simulation_manager,mcp_client
    # 项目启动时执行的命令行命令
    print("Starting up...")
    # simulation_manager = initialize_sumo(config_file=os.path.join(os.getcwd(), "sumo_llm/osm.sumocfg"),junctions_file=os.path.join(os.getcwd(), "sumo_llm/J54_data.json"))
    
    mcp_client = await get_mcp_client_async()
    
    # 启动定时任务
    task = asyncio.create_task(periodic_chat_task())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    print("Startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    global mcp_client
    print("Shutting down...")
    stop_simulation()
    if mcp_client:
        await mcp_client.cleanup()
    
    # 取消所有后台任务
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)
    
    print("Shutdown complete.")

async def call_with_messages(messages,request):
    def generate_event(content):
        """统一生成 SSE 事件数据结构"""
        return {
            "id": "chatcmpl-4",
            "choices": [{
                "delta": {
                    "content": content,
                    "function_call": "",
                    "role": "",
                    "tool_calls": ""
                },
                "finish_reason": "",
                "index": 0,
                "logprobs": ""
            }]
        }
    llm_client = LLMClient()
    while True:
        has_tool_call = False
        function_name = ""
        arguments = ""
        tool_call_id = None
        result = ""
        completion = await llm_client.get_chat_response(messages)
        # 记录当前内容类型（推理/常规）
        current_content_type = None
        for chunk in completion:
            # 原始响应数据流处理
            chunk_data = json.loads(chunk.model_dump_json())
            # print(chunk_data)
            delta = chunk_data["choices"][0].get("delta", {})
            tool_calls = delta.get("tool_calls")

            if tool_calls:
                has_tool_call = True
                # 处理每个tool_call（假设只处理第一个）
                first_tool = tool_calls[0]
                if first_tool.get("id"):
                    tool_call_id = first_tool["id"]
                
                func = first_tool.get("function", {})
                if func.get("name"):
                    function_name = func["name"]
                if func.get("arguments"):
                    arguments += func["arguments"]
            else:
                # 如果是非工具调用响应且未检测到工具调用，流式返回
                if not has_tool_call:
                    # 处理常规内容
                    if content := delta.get("content", ""):
                        if current_content_type != "content":
                            n = str("\n\n")
                            yield f'data: {json.dumps({"data": [generate_event(n)]})}\n\n'
                        current_content_type = "content"
                        yield f'data: {json.dumps({"data": [chunk_data]})}\n\n'

                    # 处理推理内容（Markdown 引用格式）
                    if reasoning_content := delta.get("reasoning_content", ""):
                        # 精确处理每行添加引用符号
                        if current_content_type is None:
                            reasoning_content = ">  " + reasoning_content
                        formatted_reasoning = reasoning_content.replace('\n','\n> ')

                        event_data = json.dumps({"data": [generate_event(formatted_reasoning)]})
                        current_content_type = "reasoning"
                        yield f'data: {event_data}\n\n'

            # 处理连接中断
            if await request.is_disconnected():
                event_data = json.dumps({"data": [generate_event("**_本次回答已被终止_**")]})
                yield f'data: {event_data}\n\n'
                break
        if has_tool_call:
            # 验证必要参数
            if not function_name or not arguments:
                print("工具调用参数不完整")

            # 构造assistant消息
            assistant_message = {
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "function": {
                        "name": function_name,
                        "arguments": arguments
                    },
                    "type": "function"
                }]
            }
            messages.append(assistant_message)
            logger.info(f"触发工具调用: {assistant_message}")

            # 执行工具调用
            mcp_client = await get_mcp_client_async()
            tool_response = await mcp_client.function_call(function_name, json.loads(arguments))
            logger.info(tool_response)
            tool_info = {"name": function_name,
                 "role": "tool",
                 "content": str(tool_response)}
            messages.append(tool_info)
            # 继续循环处理后续请求
            continue
        else:
            break

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    model = data.get("model", None)
    conversation_id = data.get("conversation_id", None)
    history_message = data.get("history", [])

    system_message = [{'role': 'system', 'content': prompt}]
    # 删除每个字典中的 'id' 键
    history_message = [{k: v for k, v in msg.items() if k != 'id'} for msg in history_message]

    junction_state_data = []
    junction_state_data.append(['倪家桥路与领事馆路交叉口,信号灯ID:J54'])
    system_message.append({'role': 'system', 'content': str(junction_state_data)})
    messages = system_message + [{'role': 'user', 'content': user_input}]
    print(messages)
    return StreamingResponse(call_with_messages(messages,request), media_type="text/event-stream")


@app.get(
    "/conversations",
    response_model='',
    operation_id="conversations",
    summary="获取会话列表"
)
async def select_conversations():
    return []

@app.get(
    "/conversations/{conversation_id}",
    response_model='',
    operation_id="conversationsMessagesList",
    summary="获取会话的详细对话记录"
)
async def select_conversation_messages():
    return []

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}

def start():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)

if __name__ == "__main__":
    start()


