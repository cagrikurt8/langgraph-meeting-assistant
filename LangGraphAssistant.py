from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, BaseMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
import os


#########################################
# TOOLS #
#########################################
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


#########################################
# State Class #
#########################################
class State(MessagesState):
    user_id: str


#########################################
# Assistant Class #
#########################################
class LangGraphAssistant:
    def __init__(self, user_id):
        load_dotenv()
        self.user_id = user_id
        #self.mongodb_saver = MongoDBSaver.from_conn_string(os.getenv("MONGODB_URI"))
        self.memory_saver = MemorySaver()
        self.llm = AzureChatOpenAI(azure_deployment="HayatAI-GPT4o", api_version="2024-10-21", temperature=0)
        self.tools = [add, multiply, divide]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.sys_msg = SystemMessage(content=open("system_message.txt", "r").read())
        self.thread_id = {"configurable": {"thread_id": self.user_id}}
        self.graph = self.build_graph()
    

    def build_graph(self):
        # Graph
        builder = StateGraph(State)
        
        # Define nodes: these do the work
        builder.add_node("assistant", self.call_model)
        builder.add_node("tools", ToolNode(self.tools))

        # Define edges: these determine the control flow
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        graph = builder.compile(checkpointer=self.memory_saver)

        return graph


    def call_model(self, state: State, config: RunnableConfig):
        return {"messages": self.llm_with_tools.invoke([self.sys_msg] + state["messages"], config)}

    
    def stream_answer(self, question):
        input_message = HumanMessage(content=question)

        return self.graph.astream_events({"messages": [input_message]}, self.thread_id, version="v2")


    def get_answer(self, question):
        input_message = HumanMessage(content=question)
   
        return self.graph.invoke({"messages": [input_message]}, self.thread_id)
    
    
    def get_agent_state(self):
        return self.graph.get_state(self.thread_id)