import os

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))

# LangChain 内置了一个工具，可以轻松调用Tavily搜索引擎作
search = TavilySearchResults(max_results=2)  # 回搜索很多数据，可以给定最多的返回结果

tools = [search]

# 让模型绑定工具
model_with_tool = model.bind_tools(tools)  # 参数是sequence

# 模型可以自动推理 是否需要调用工具去完成用户的答案
resp = model_with_tool.invoke([HumanMessage(content='中国的首都是哪个城市？')])
print(f'Model Result Content: {resp.content}')
print(f'Model Result Content: {resp.tool_calls}')  # 不需要模型工具去搜索，所以输出为空

resp1 = model_with_tool.invoke([HumanMessage(content='今天大连的天气怎么样？')])
print(f'Model Result Content: {resp1.content}')  # 模型不知道，返回为空
print(f'Model Result Content: {resp1.tool_calls}')  # 模型需要搜索，给出的是搜索的指令，可以去调用工具，此时还没有去调用工具

# 创建代理

agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

resp= agent_executor.invoke({'messages':[HumanMessage(content='中国的首都是哪个城市？')]})
print(resp['messages'])
resp1= agent_executor.invoke({'messages':[HumanMessage(content='今天大连的天气怎么样？')]})
print(resp1['messages']) # 返回值有 HumanMessage 用户问的信息，AIMessage 用户调用工具的信息，ToolMessage search工具搜索的结果
print(resp1['messages'][2].content)# ToolMessage 的返回结果
