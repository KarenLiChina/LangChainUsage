## python环境要求
```bash
pip install -r requirements.txt
```

## 配置环境变量

创建 `.env`文件，在env文件中的设置 `OPENAI_API_KEY` 和 `BASE_URL` 为自己的 key 和 url
`LANGCHAIN_TRACING_V2`设置为true，`LANGCHAIN_API_KEY`设置为LangSmith的API Key，可以在LangSmith中查看调用大模型使用情况，不需要也可以不配置这两个变量

# langsmith的检测数据
配置`LANGCHAIN_TRACING_V2`，`LANGCHAIN_API_KEY`后可以在https://smith.langchain.com/ Tracing Projects中查看调用大模型的使用情况


## 用langchain 链式调用LLM 案例，引入LangSmithAPIkey，就可以用LangSmith去追踪
call_openai.py

## 使用prompt template 去调用，并且用FastAPI进行部署
call_LLM_withPromptTemplate.py

测试curl：
```bash
curl -X POST http://localhost:8000/translate_ai/invoke -H "Content-Type: application/json" -d '{"input":{"language":"英语","content":"明月几时有，把酒问青天。"}}'
```