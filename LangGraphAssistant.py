from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.runnables import RunnableConfig
import os
from AssistantFunctions import *
from pymongo import MongoClient
from datetime import datetime


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
        self.user_id = user_id
        self.mongodb_saver = MongoDBSaver(MongoClient(os.getenv("MONGODB_URI")))
        #self.memory_saver = MemorySaver()
        self.llm = AzureChatOpenAI(azure_deployment=os.getenv("MODEL_NAME"), api_version="2024-10-21", temperature=0)
        #self.python_repl = SessionsPythonREPLTool(pool_management_endpoint=os.getenv("POOL_MANAGEMENT_ENDPOINT"))
        self.tools = [add, multiply, divide, web_search, python_repl]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.sys_msg = SystemMessage(content=open("system_message.txt", "r").read() + f" Today is {datetime.now().strftime('%Y-%m-%d')}, in case you need the date to complete your tasks.")
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
        
        graph = builder.compile(checkpointer=self.mongodb_saver)
        #graph = builder.compile(checkpointer=self.memory_saver)

        return graph


    def call_model(self, state: State, config: RunnableConfig):
        return {"messages": self.llm_with_tools.invoke([self.sys_msg] + state["messages"], config)}

    
    def stream_answer(self, question):
        input_message = HumanMessage(content=question)

        return self.graph.stream({"messages": [input_message]}, self.thread_id, stream_mode="messages")


    def get_answer(self, question):
        input_message = HumanMessage(content=question)
   
        return self.graph.invoke({"messages": [input_message]}, self.thread_id)
    
    
    def get_agent_state(self):
        return self.graph.get_state(self.thread_id)