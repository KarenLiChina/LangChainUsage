import os

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"))

# sqlalchemy 连接数据库，需要安装要连接的数据库驱动库，初始化MySQL数据库的连接，参数配到.env 文件中
HOSTNAME = os.getenv("MYSQL_HOSTNAME")
PORT = os.getenv("MYSQL_PORT")
DATABASE = os.getenv("MYSQL_DATABASE")
USERNAME = os.getenv("MYSQL_USERNAME")
PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db_connection = SQLDatabase.from_uri(MYSQL_URI)

# 创建工具
tool_kit = SQLDatabaseToolkit(db=db_connection, llm=model)
tools = tool_kit.get_tools()
# 使用Agent 完成整个数据库的整合

system_prompt = """
你是一个SQL数据交互的代理。给定一个输入问题，创建一个语法正确的SQL语句并执行，然后擦看查询结果并返回答案。
除非用户给定了他们想要获得的具体数量，否则始终将查询现在为最多10个结果。
你可以按照相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
你可以使用数据库交互工具，在执行查询之前，你必须自己想检测，如果在执行查询是出现错误，请重新查询并重试。
不要多数据库进行任何DML语句（插入、更新、删除等）。
"""
system_message = SystemMessage(content=system_prompt)

# 创建代理

# 将系统消息和工具准备好，新版本创建agent的方法在langchain.agents 下
agent_executor = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_message  # 直接传递系统提示文本
)

resp = agent_executor.invoke({'messages': [HumanMessage(content='那种性别的员工人数最多？')]})

result = resp['messages']
print(result)
print(len(result))
# 最后一个才是真正的答案
print(result[len(result)-1])