import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))


# pydantic: 处理数据，验证数据，定义数据格式，虚拟化和反虚拟化，类型转换等等

# 定义数据模型

class Person(BaseModel):
    """
    关于一个人的数据模型
    """
    name: Optional[str] = Field(default=None, description='人物的名字')
    coat_color: Optional[str] = Field(default=None, description='人物衣服的颜色')
    age: Optional[str] = Field(default=None, description='人物的年纪')


# 定义自定义提示词以供指令和其他人任何上下文。
# 1）可以在提示词每一步中添加示例以提高提取质量 2）引入额外的参数以考虑上下文
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的提取算法。只给未结构化文本中提取相关信息。如果你不知道要提取的属性值，返回该属性的值为null。"),
    ("human", "{text}")
])

# model.with_structured_output 表示模型的输出是一个结构化的数据
# 当pydantic.v1 时，不需要加 method="function_calling",当使用pydantic 即v2 时，需要显示加 method="function_calling"
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=Person, method="function_calling")

result = chain.invoke('只见船尾一个女子持浆荡舟，长发披肩，全身白衣，头发上束了条金带，白雪一映，更是灿然生光。郭靖见这少女一身装束犹如仙女一般，不禁看的呆了。那船慢慢荡近，只见这女子方当韶龄，不过十五六岁年纪，肌肤胜雪、娇美无匹；容色绝丽，不可逼视。')

print(result)
