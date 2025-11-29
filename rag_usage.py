import os

import bs4
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 创建模型
model = ChatOpenAI(model='gpt-4-turbo',
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))

# 1. 加载数据
loader = WebBaseLoader(web_paths=['https://blog.csdn.net/2301_82275412/article/details/148773003'],
                       # 参数可以是一个路径，也可以是路径数组
                       bs_kwargs=dict(
                           parse_only=bs4.SoupStrainer("article")  # bs4是解析HTML的解析器，根据解析的html文件定义解析的参数
                       )
                       )
docs = loader.load()

# 2. 大文本的切割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splits = splitter.split_documents(docs)

# 3. 存储
vectorstore = Chroma.from_documents(splits, embedding=OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")))

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# 5. 整合
# 创建问题的prompt模板
system_prompt = """你是个可靠的专门做问答的助手，用下面的文本回答问题，如果你不知道答案，回答不知道，最多用三句话来准确回答，保证答案的简洁：
{context}
"""

prompt = ChatPromptTemplate.from_messages([  # 提问和问答历史记录的模板
    ("system", system_prompt),
    # MessagesPlaceholder("chat_history"),  # 提问的历史记录
    ("human", "{input}")
])

# 通过函数create_stuff_documents_chain创建链chain
answer_chain = create_stuff_documents_chain(model, prompt)  # 只有问答的chain

retriever_chain = create_retrieval_chain(retriever, answer_chain)  # 包括检索器的chain

resp = retriever_chain.invoke({'input': '金融行业用AI做什么？'})
print(resp['answer'])
