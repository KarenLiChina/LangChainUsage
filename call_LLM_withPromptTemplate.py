import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

load_dotenv()

# 调用LLM
# 1. 创建模型

model = ChatOpenAI(model='gpt-4-turbo',
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))
# 2. 定义提示词模板 prompt
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译成{language}'),
    ('user', '{content}')
])

# 3. 创建返回数据的解析器
parser = StrOutputParser()  # langchain中最简单的解析工具

# 4. 构建链
chain = prompt_template | model | parser

# 5. 直接使用chain调用
#print(chain.invoke({'language': '英语', 'content': '明月几时有，把酒问青天。'}))  #

# 把程序部署成服务，可以让其他人去调用
# 创建fastAPI的应用

app=FastAPI(title='Translate AI', version='v1.0',description='使用AI翻译语句的服务')
add_routes(
    app,
    chain,
    path='/translate_ai'
)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)