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
from datetime import datetime, timedelta
from typing import List
import base64

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
    def __init__(self, thread_id, user_id, python_repl):
        self.thread_id = thread_id
        self.user_id = user_id
        self.file = None
        self.python_repl = python_repl
        self.mongodb_saver = MongoDBSaver(MongoClient(os.getenv("MONGODB_URI")))
        #self.memory_saver = MemorySaver()
        self.llm = AzureChatOpenAI(azure_deployment=os.getenv("MODEL_NAME"), api_version="2024-10-21", temperature=0)
        #self.llm = BaseChatOpenAI(model='deepseek-chat', openai_api_key=os.getenv("DEEPSEEK_API_KEY"), openai_api_base='https://api.deepseek.com', max_tokens=1024, temperature=0)
        #self.python_repl = SessionsPythonREPLTool(pool_management_endpoint=os.getenv("POOL_MANAGEMENT_ENDPOINT"))
        self.tools_by_name = {"add": add, "multiply": multiply, "divide": divide, "web_search": web_search, "Python_REPL": self.python_repl, "get_all_meetings": get_all_meetings, "get_meeting_transcript_contents": get_meeting_transcript_contents}
        self.tools = [add, multiply, divide, web_search, self.python_repl, get_all_meetings, get_meeting_transcript_contents]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.sys_msg = open("system_message.txt", "r").read() + f" Today is {datetime.now().strftime('%Y-%m-%d')}, in case you need the date to complete your tasks. The user ID of the user is {user_id}, you can use it in your tools."
        self.thread_id = {"configurable": {"thread_id": self.thread_id, "user_id": self.user_id}}
        self.graph = self.build_graph()


    def build_graph(self):
        # Graph
        builder = StateGraph(State)
        
        # Define nodes: these do the work
        builder.add_node("assistant", self.call_model)
        #builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("tools", self.tool_node)
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
        idx = 0

        while isinstance(trimmed_messages[idx], ToolMessage):
            idx += 1

        trimmed_messages = trimmed_messages[idx:]
    
        if summary:
            summary_message = f"This is summary of the conversation to date: {summary}\n\n" \
                               "Extend the summary by taking into account the new messages above:"
            
        else:
            summary_message = "Create a summary of the conversation above:"
        
        messages = [SystemMessage(content=self.sys_msg)] + trimmed_messages + [HumanMessage(content=summary_message)]
        response = self.llm_with_tools.invoke(messages, config)
        
        return {"summary": response.content}


    def tool_node(self, state: State, config: RunnableConfig):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            try:
                if tool_call["name"] == "Python_REPL":
                    observation = self.python_repl.execute(tool_call["args"]["python_code"])
                 
                    if isinstance(observation['result'], dict) and observation["result"]["type"]== 'image':
                        result.append(ToolMessage(content="Code execution is successfull and the image is already displayed on the UI. You can inform the user accordingly.", name=tool_call["name"], artifact=observation, tool_call_id=tool_call["id"]))
                    else:
                        result.append(ToolMessage(content=str(observation), name=tool_call["name"], tool_call_id=tool_call["id"]))
                elif tool_call["name"] == "get_all_meetings":
                    observation = tool.invoke(tool_call["args"])
                    result.append(ToolMessage(content=str(observation[0]), name=tool_call["name"], tool_call_id=tool_call["id"]))
                else:
                    observation = tool.invoke(tool_call["args"])
                    result.append(ToolMessage(content=str(observation), name=tool_call["name"], tool_call_id=tool_call["id"]))
                
            except Exception as e:
                print(f"Error: {e}")
                result.append(ToolMessage(content=f"Error: {e}", name=tool_call["name"], tool_call_id=tool_call["id"]))
        
        return {"messages": result}


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
            
            for idx, m in enumerate(trimmed_messages):
                if isinstance(m, ToolMessage) and m.name == "python_repl":
                    try:
                        result_dict = json.loads(m.content)
                        if result_dict['result']['type'] == 'image':
                            trimmed_messages[idx].content = "Image generation is successfull."
                            trimmed_messages[idx].artifact = json.dumps(result_dict)
                    except:
                        continue

            messages = [SystemMessage(content=f"{self.sys_msg} {summary_msg}")] + trimmed_messages
        else:
            for idx, m in enumerate(state["messages"]):
                if isinstance(m, ToolMessage) and m.name == "python_repl":
                    try:
                        result_dict = json.loads(m.content)
                        if result_dict['result']['type'] == 'image':
                            state["messages"][idx].content = "Image generation is successfull."
                            state["messages"][idx].artifact = json.dumps(result_dict)
                    except:
                        continue

            messages = [SystemMessage(content=self.sys_msg)] + state["messages"]
        print("Messages to be sent:")
        print(messages)
        
        return {"messages": self.llm_with_tools.invoke(messages, config), "user_id": self.user_id}

    
    def stream_answer(self, question):
        input_message = HumanMessage(content=question)

        return self.graph.stream({"messages": [input_message]}, self.thread_id, stream_mode="messages")


    def set_file(self, file):
         self.file = file
         self.python_repl.upload_file(data=BytesIO(self.file.read()), remote_file_path=self.file.name)
         self.sys_msg += f" The file is set, you can use it in your tools. The file name is '/mnt/data/{file.name}'."


    def get_answer(self, question):
        input_message = HumanMessage(content=question)
   
        return self.graph.invoke({"messages": [input_message]}, self.thread_id)
    
    
    def get_agent_state(self):
        return self.graph.get_state(self.thread_id)