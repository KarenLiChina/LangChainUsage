import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# 调用LLM
# 1. 创建模型

model = ChatOpenAI(model='gpt-4-turbo',
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))
# 2. 准备提示词 prompt
msg = [
    SystemMessage(content='请将一下的内容翻译为英语'),
    HumanMessage(content="明月几时有，把酒问青天。")
]


# 3. 创建返回数据的解析器
parser = StrOutputParser()  # langchain中最简单的解析工具

# 4. 得到链,链式调用时，每一项必须是Runnable的对象
chain = model | parser

# 5. 直接使用chain调用
print(chain.invoke(msg)) #

# 链式调用等价与下面的代码
# result = model.invoke(msg)
# print(parser.invoke(result)) # 解析相应之后的结果
