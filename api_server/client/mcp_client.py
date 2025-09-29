import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

# from .llm_client import LLMClient



class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._loop = None
        # self.llm_client = LLMClient(model_type="openai")

    def _ensure_loop(self):
        """确保事件循环存在"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def sync_list_tools(self):
        """同步方式获取工具列表"""
        loop = self._ensure_loop()
        return loop.run_until_complete(self.get_list_tools())

    def sync_call_tool(self, tool_name: str, tool_args: dict):
        """同步方式调用工具"""
        loop = self._ensure_loop()
        return loop.run_until_complete(self.session.call_tool(tool_name, tool_args))

    def sync_process_query(self, query: str) -> str:
        """同步方式处理查询"""
        loop = self._ensure_loop()
        return loop.run_until_complete(self.process_query(query))

    def sync_cleanup(self):
        """同步方式清理资源"""
        loop = self._ensure_loop()
        return loop.run_until_complete(self.cleanup())

    async def connect_to_server(self, server_path: str, transport_type: str = "stdio"):
        """Connect to an MCP server
        
        Args:
            server_path: Path to the server script (.py or .js) or URL for SSE connection
            transport_type: Type of transport ('stdio' or 'sse')
        """
        if transport_type == "stdio":
            # 使用stdio方式连接
            is_python = server_path.endswith('.py')
            is_js = server_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")
                
            command = "python" if is_python else "node"
            print(f"Starting server with {command}")
            server_params = StdioServerParameters(
                command=command,
                args=[server_path],
                env=None
            )
            print(server_params)
            transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        
        elif transport_type == "sse":
            # 使用SSE方式连接
            print(f"Connecting to SSE server at {server_path}")
            transport = await self.exit_stack.enter_async_context(sse_client(server_path))
        
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
        
        self.stdio, self.write = transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        # 初始化会话
        await self.session.initialize()
        

    async def get_list_tools(self):
        """获取可用工具列表"""
        res = await self.session.list_tools()
        tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
                }
        } for tool in res.tools]
        return tools
    
    async def function_call(self, function_name: str, function_args: dict):
        """执行工具调用"""
        print(function_name, function_args)
        res = await self.session.call_tool(function_name, function_args)
        print(res)
        return res

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
                }
        } for tool in response.tools]
        
        response = await self.llm_client.get_completion(
            messages=messages,
            model='qwen-max',
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []
        print(response)
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from LLM
                response = await self.llm_client.get_completion(
                    messages=messages,
                    model='qwen-max'
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# 全局实例
_mcp_client: Optional[MCPClient] = None

def get_mcp_client() -> Optional[MCPClient]:
    """获取全局MCP客户端实例"""
    global _mcp_client
    if _mcp_client is None:
        try:
            # 创建并初始化客户端
            _mcp_client = MCPClient()
            loop = _mcp_client._ensure_loop()
            
            # 初始化连接
            loop.run_until_complete(_mcp_client.connect_to_server("http://0.0.0.0:8000/sse", "sse"))
            loop.run_until_complete(_mcp_client.session.initialize())
            
            # 验证连接
            tools = loop.run_until_complete(_mcp_client.get_list_tools())
            if tools:
                print(f"成功获取到 {len(tools)} 个工具")
                print("\nConnected to server with tools:", [tool["function"]["name"] for tool in tools])
            else:
                print("警告: 未获取到工具列表")
        except Exception as e:
            print(f"初始化MCP客户端时发生错误: {str(e)}")
            
            raise
    return _mcp_client

async def get_mcp_client_async() -> Optional[MCPClient]:
    """获取全局MCP客户端实例"""
    global _mcp_client
    if _mcp_client is None:
        try:
            print("开始初始化MCP客户端...")  # 调试日志
            _mcp_client = MCPClient()
            
            # 初始化连接
            await _mcp_client.connect_to_server("http://0.0.0.0:8000/sse", "sse")
            print("连接服务器成功")  # 调试日志
            
            # 验证连接
            tools = await _mcp_client.get_list_tools()
            if tools:
                print(f"成功获取到 {len(tools)} 个工具")
                print("\nConnected to server with tools:", [tool["function"]["name"] for tool in tools])
            else:
                print("警告: 未获取到工具列表")
        except Exception as e:
            print(f"初始化MCP客户端时发生错误: {str(e)}")
            _mcp_client = None
            raise
    return _mcp_client
