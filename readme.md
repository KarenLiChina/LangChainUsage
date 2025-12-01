## python环境要求
```bash
pip install -r requirements.txt
```

## 配置环境变量

创建 `.env`文件，在env文件中的设置 `OPENAI_API_KEY` 和 `BASE_URL` 为自己的 key 和 url
`LANGCHAIN_TRACING_V2`设置为true，`LANGCHAIN_PROJECT`设置为项目名称，不配置默认为default，`LANGCHAIN_API_KEY`设置为LangSmith的API Key，可以在LangSmith中查看调用大模型使用情况，不需要也可以不配置这两个变量

# langsmith的检测数据
配置`LANGCHAIN_TRACING_V2`，`LANGCHAIN_API_KEY`后可以在https://smith.langchain.com/ Tracing Projects中查看调用大模型的使用情况

# Tavily API Key的配置
配置`TAVILY_API_KEY` 为你自己的API key，可以实现搜索功能，在agent_with_search.py 代码中用到
可以在 https://app.tavily.com/home 申请
## 用langchain 链式调用LLM 案例，引入LangSmithAPIkey，就可以用LangSmith去追踪
链式调用时，每一项必须是Runnable的对象
call_openai.py

## 使用prompt template 去调用，并且用FastAPI进行部署
call_LLM_withPromptTemplate.py

测试curl：
```bash
curl -X POST http://localhost:8000/translate_ai/invoke -H "Content-Type: application/json" -d '{"input":{"language":"英语","content":"明月几时有，把酒问青天。"}}'
```

## 构建聊天机器人，使其能够对话并记住之前的互动（chat History），流式输出，依赖 langchain_community包
aichatbot.py

## 从向量数据库中检索数据，基于自己的数据作为大模型推理的一部分，依赖langchain-chroma， chroma是langchain内部提供的向量数据库
vector_data_rag.py

# 构建代理，大语言模型本身无法执行动作，只能输出文本，大语言模型可以通过推理确定要执行的操作，以及这些操作的输出，有代理决定是否需要更多的操作。依赖langgraph
agent_with_search.py

# RAG 是一种增强大语言模型LLM知识的方法，它通过引入额外的数据来实现
具体实现步骤： 加载-》分割-》存储-》检索-》生成
检索器也需要上下文来进行检索，需要构建一个子链，采用用户最新的问题和历史聊天记录，让检索过程融入对话上下文。
rag_usage.py