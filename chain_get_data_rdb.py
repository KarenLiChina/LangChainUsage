import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

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

# 直接使用大模型和数据库整合，只能能够根据你的问题生成SQL
sql_chain = create_sql_query_chain(model, db_connection)
# resp = sql_chain.invoke({'question':'请问用户表中有多少条数据？'}) # invoke的参数是字典
# print(resp) # 此时的输出结果为SQLQuery: SELECT COUNT(*) AS `count` FROM users;

# 定义根据用户问题，获得的SQL语句和SQL的结果来回答问题的提示词模板
answer_prompt = PromptTemplate.from_template(
    """给定一下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question:{question}
    SQL Query:{query}
    SQL Result:{result}
    回答：
    """
)
# 创建执行SQL语句的工具
execute_sql_tool = QuerySQLDatabaseTool(db=db_connection)  #

# 1. 生成SQL 语句，2. 执行SQL
# RunnablePassthrough 创建一个可以执行的对象, query对应提示词中的query，result对应提示词中等result，itemgetter 得到query执行的结果
# 使用chain的方式时，代码比较复杂，比较新的大模型返回结果都回有 "SQLQuery:"前缀，导致sql无法用 db的tool去执行，GPT3.5 版本可以生成没有前缀的SQL
chain = (RunnablePassthrough.assign(query=sql_chain).assign(
    result=itemgetter('query') | execute_sql_tool)
         | answer_prompt
         | model
         | StrOutputParser())

resp = chain.invoke({'question':'请问用户表中有多少条数据？'})
print(resp)

