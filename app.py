import os
import certifi


from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv


from langchain.tools import tool

# ==========================================
# PROMPT
# ==========================================

prompt = """Answer the user's question as best you can.

You have access to tools that you can use to gather information.
Use the available tools when necessary, observe their results,
and continue until you have enough information to answer.

Never claim that information came from the database unless
the tool actually returned that information.

Give a clear final answer based on the information you gathered."""


# ===========================================
# Set up environment variables
# ============================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()


OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
TAVILY_API_KEY = os.getenv("Tavily_API_KEY")


# https://docs.ollama.com/api/openai-compatibility
# https://reference.langchain.com/python/langchain-openai/langchain_openai/chat_models/base/ChatOpenAI
llm = ChatOpenAI(
    model="gpt-oss:120b",
    base_url="https://ollama.com/v1",
    api_key=os.environ["OLLAMA_API_KEY"],
    temperature=0,
)

search_tool = TavilySearch(max_results=3)

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    print(f"Searching database for '{query}' with limit {limit}...")

    return (
            "Database unavailable: this is a demo tool and "
            "no real database is connected. No records were retrieved."
        )

# ==========================================
# TOOLS
# ==========================================

tools = [search_tool, search_database]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt
)

# https://docs.langchain.com/oss/python/langchain/messages
# response = agent.invoke({
#     "messages": "Find the capital of India and its current weather and what we had for its in database"
# })

# print(response["messages"][-1].content)



import streamlit as st

st.title("My AI Agent")
st.write("Ask a question and let the agent find an answer.")

question = st.text_input(
    "Your question",
    placeholder="What is the capital of India?"
)

if st.button("Ask") and question:
    with st.spinner("Agent is working..."):
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

    answer = response["messages"][-1].content
    st.markdown(answer)
