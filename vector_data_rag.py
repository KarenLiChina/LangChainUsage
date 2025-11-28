import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# 调用LLM
# 1. 创建模型

model = ChatOpenAI(model='gpt-4-turbo',
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))

# 准备测试数据, animals.txt 中每一行作为Document的page_content 即为文本内容，metadata是文档的一些元数据，是键值形式
file_path = './data/animals.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    documents = [
        Document(
            page_content=line.strip(),
            metadata={"source": "animals.txt"}
        )
        for line in file if line.strip()  # 跳过空行
    ]

# 实例化一个向量数据库
vector_store = Chroma.from_documents(documents,
                                     embedding=OpenAIEmbeddings(
                                         api_key=os.getenv("OPENAI_API_KEY"),
                                         base_url=os.getenv("BASE_URL")))  # 使用OpenAIEmbedding 工具来进行向量化

# 相似度查询：返回相似度的分数，Chroma中是分数越低，相似度越高,分数表示距离
# print(vector_store.similarity_search_with_score('黑白猫'))

# 根据向量空间得到一个检索器
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)  # 返回相似度最高的一个

# 定义提示模板
message = """
使用提供的上下文回答这个问题：
{question}
上下文:
{content}
"""
prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# RunnablePassthrough 允许将用户的提问之后传递给prompt和model
chain = {'question': RunnablePassthrough(), 'content': retriever} | prompt_temp | model
resp = chain.invoke('介绍一下黑白猫？')
print(resp.content)
