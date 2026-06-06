from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
prompt = Client().pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True
)

generation_chain = prompt | llm | StrOutputParser()
