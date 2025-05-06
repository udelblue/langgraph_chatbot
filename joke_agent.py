import dotenv

from langchain_openai import ChatOpenAI

from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_core.messages import trim_messages

from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# Load api keys from .env file.
dotenv.load_dotenv()

# Extra fields in state could be added here.
class State(MessagesState):
    pass

# Set up the finest jokes of all time.
@tool
def get_joke_setup():
    """Get some joke setups"""
    return ["Why is a pirate's favorite letter 'R'?", "What do you call a fish without eyes?"]

@tool
def get_joke_punchline():
    """Get some joke punchlines"""
    return ["Because, if you think about it, 'R' is the only letter that makes sense.",
            "Its a fsh!"]

tools = [get_joke_setup, get_joke_punchline]

def make_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    # Limit the size of the context window provided to the LLM.
    trimmer = trim_messages(
        max_tokens=10000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human")
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
            You are a expert at telling jokes.
            You always start with the setup, and wait for the user to respond before telling the punch line.
            Prefer using tools to get the joke setup and punchline.
            Example:
            user: tell me a joke!
            ai: why did the chicken cross the road?
            user: why?
            ai: to get to the other side!
            """),
        MessagesPlaceholder(variable_name="messages")])

    chain = trimmer | prompt_template | model.bind_tools(tools)

    def agent(state: State):
        print("---- jokes agent ----")
        response = chain.invoke(state["messages"])
        print("---- jokes agent finished ----")
        return {"messages": response}

    return agent

# A basic tool using agent setup.
# This setup is almost identical to LangGraph's create_react_agent function.
graph = StateGraph(state_schema=State) \
    .add_node("agent", make_agent()) \
    .add_node("tools", ToolNode(tools)) \
    .set_entry_point("agent") \
    .add_conditional_edges("agent", tools_condition) \
    .add_edge("tools", "agent") \
    .compile(checkpointer=InMemorySaver())