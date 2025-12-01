import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# 调用LLM
# 1. 创建模型

model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))
# 2. 定义提示词模板 prompt
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个非常聪明的助手，用{language}尽你所能回答所有问题。'),
    MessagesPlaceholder(variable_name='chat_msg') # 与input_messages_key 保持一致
])

# 定义链
chain = prompt_template | model

# 保存聊天的历史记录

store = {}  # 所有用户的聊天记录都保存到store中。Key为sessionId， value为历史聊天记录对象


# 回调函数，收到session_id并返回一个消息历史记录对象
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,  # 回调函数，不要加括号
    input_messages_key='chat_msg'  # 每次聊天时候发msg的key
)

config = {'configurable': {'session_id': 'abc123'}}  # 给当前用户定义一个session_id
# 第一轮发的消息
result = do_message.invoke(
    {
        'chat_msg': [HumanMessage(content='你好，我是Karen。')],# 与input_messages_key 保持一致
        'language': '中文'
    },
    config=config
)

print(result.content)
# 第二轮发的消息
result2 = do_message.invoke(
    {
        'chat_msg': [HumanMessage(content='请问我的名字是什么？')],# 与input_messages_key 保持一致
        'language': '中文'
    },
    config=config
)
print(result2.content)

# 第三轮: 返回的数据是流式的，方法就式stream,循环输出
for resp in do_message.stream(
    {
        'chat_msg': [HumanMessage(content='请给我讲个故事')],# 与input_messages_key 保持一致
        'language': '英文'
    },
    config={'configurable': {'session_id': 'def345'}}  # 换一个session_id,开启一轮新的对话
):
    # 每一次resp都是一个token
    print(resp.content,end='-')