import os

import bs4
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
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
    MessagesPlaceholder("chat_history"),  # 提问的历史记录
    ("human", "{input}")
])

# 通过函数create_stuff_documents_chain创建链chain
answer_chain = create_stuff_documents_chain(model, prompt)  # 只有问答的chain

'''
一般情况下，我们构建的链（chain）直接使用输入问答记录来关联上下文，在此案例中，不仅是模型，检索器也需要对话的上下文才能更好的理解。
解决方法：
添加一个子链，它采用最新用户问题和聊天历史，并在它引用历史信息中的任何信息是重新表述问题，这样可以被简单地任务是构建一个新的“历史感知”检索器。
子链的目的：让检索过程融入对话上下文。
'''

# 先创建子链的提示词模板
contextualize_q_system_prompt = """
你负责根据对话历史和当前问题生成一个独立、完整的问题。请遵循以下规则：
1. 当存在历史对话记录时，结合上下文和当前问题，生成一个语义完整、可独立理解的新问题
2. 当没有历史对话记录时，直接转述或返回原始问题本身
3. 确保生成的问题保持原意，语言自然流畅"""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# 创建一个子链
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# 保持问答的历史记录
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建一个父链（chain） 将检索器的链和问答链，两个链组合起来

chain = create_retrieval_chain(history_chain, answer_chain)  # 包括检索器的chain

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)
# 第一轮提问
resp1 = result_chain.invoke(
    {'input': '金融行业用AI做什么？'},
    config={'configurable': {'session_id': 'abc1234'}}
)
print(resp1['answer'])

resp2 = result_chain.invoke(
    {'input': '可以选择哪种大模型？'},
    config={'configurable': {'session_id': 'abc1234'}}
)
print(resp2['answer'])
