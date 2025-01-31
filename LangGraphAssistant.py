from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_openai import AzureChatOpenAI
#from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, RemoveMessage, BaseMessage, trim_messages
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.runnables import RunnableConfig
import os
from AssistantFunctions import *
from pymongo import MongoClient
from datetime import datetime
from typing import List


#########################################
# State Class #
#########################################
class State(MessagesState):
    user_id: str
    summary: str
    #trimmed_messages: List[BaseMessage]


#########################################
# Assistant Class #
#########################################
class LangGraphAssistant:
    def __init__(self, thread_id, user_id):
        self.thread_id = thread_id
        self.user_id = user_id
        self.mongodb_saver = MongoDBSaver(MongoClient(os.getenv("MONGODB_URI")))
        #self.memory_saver = MemorySaver()
        self.llm = AzureChatOpenAI(azure_deployment=os.getenv("MODEL_NAME"), api_version="2024-10-21", temperature=0)
        #self.llm = BaseChatOpenAI(model='deepseek-chat', openai_api_key=os.getenv("DEEPSEEK_API_KEY"), openai_api_base='https://api.deepseek.com', max_tokens=1024, temperature=0)
        #self.python_repl = SessionsPythonREPLTool(pool_management_endpoint=os.getenv("POOL_MANAGEMENT_ENDPOINT"))
        self.tools = [add, multiply, divide, web_search, python_repl, get_all_meetings, get_meeting_transcript_contents]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.sys_msg = open("system_message.txt", "r").read() + f" Today is {datetime.now().strftime('%Y-%m-%d')}, in case you need the date to complete your tasks. The user ID of the user is {user_id}, you can use it in your tools."
        self.thread_id = {"configurable": {"thread_id": self.thread_id, "user_id": self.user_id}}
        self.graph = self.build_graph()


    def build_graph(self):
        # Graph
        builder = StateGraph(State)
        
        # Define nodes: these do the work
        builder.add_node("assistant", self.call_model)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("summarize_conversation", self.summarize_conversation)

        # Define edges: these determine the control flow
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            self.should_continue,
            #tools_condition
            ["tools", "summarize_conversation", END]
        )
        builder.add_edge("tools", "assistant")
        builder.add_edge("summarize_conversation", END)
        
        graph = builder.compile(checkpointer=self.mongodb_saver)
        #graph = builder.compile(checkpointer=self.memory_saver)

        return graph


    def should_continue(self, state: State, config: RunnableConfig):
        messages = state["messages"]

        if len(messages) > 6:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                        return "tools"
                return "summarize_conversation"
                
        else:
               last_message = messages[-1]
               if isinstance(last_message, AIMessage) and last_message.tool_calls:
                       return "tools"
               return END
        

    def summarize_conversation(self, state: State, config: RunnableConfig):
        summary = state.get("summary", "")
        trimmed_messages = trim_messages(
                                state["messages"][-12:],
                                max_tokens=6,
                                strategy="first",
                                token_counter=len,
                                allow_partial=False,
                                #start_on="human",
                                end_on=("human", "tool")
                            )
        if isinstance(trimmed_messages[0], ToolMessage):
             trimmed_messages = trimmed_messages[1:]
        print("Summarize Trimmed Messages:")
        for m in trimmed_messages:
             m.pretty_print()
        print()
        print()
        if summary:
            summary_message = f"This is summary of the conversation to date: {summary}\n\n" \
                               "Extend the summary by taking into account the new messages above:"
            
        else:
            summary_message = "Create a summary of the conversation above:"
        
        messages = [SystemMessage(content=self.sys_msg)] + trimmed_messages + [HumanMessage(content=summary_message)]
        response = self.llm_with_tools.invoke(messages, config)
        
        return {"summary": response.content}


    def call_model(self, state: State, config: RunnableConfig):
        summary = state.get("summary", "")

        if summary:
            summary_msg = f"Summary of conversation earlier: {summary}"
            trimmed_messages = trim_messages(
                                state["messages"],
                                max_tokens=6,
                                strategy="last",
                                token_counter=len,
                                allow_partial=False,
                                start_on="human",
                                end_on=("human", "tool")
                            )
            messages = [SystemMessage(content=f"{self.sys_msg} {summary_msg}")] + trimmed_messages
        else:
            messages = [SystemMessage(content=self.sys_msg)] + state["messages"]
        print("Messages to be sent:")
        print(messages)
        
        return {"messages": self.llm_with_tools.invoke(messages, config), "user_id": self.user_id}

    
    def stream_answer(self, question):
        input_message = HumanMessage(content=question)

        return self.graph.stream({"messages": [input_message]}, self.thread_id, stream_mode="messages")


    def get_answer(self, question):
        input_message = HumanMessage(content=question)
   
        return self.graph.invoke({"messages": [input_message]}, self.thread_id)
    
    
    def get_agent_state(self):
        return self.graph.get_state(self.thread_id)