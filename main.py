from LangGraphAssistant import LangGraphAssistant
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool


st.set_page_config(
    page_title="LangGraph Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)


def stream_response(stream):
    for token in stream:
        if isinstance(token[0], AIMessage) and len(token[0].content) > 0 and token[1]["langgraph_node"] == "assistant":
            yield token[0].content
        
        elif isinstance(token[0], ToolMessage) and token[0].artifact:
            try:
                result_dict = token[0].artifact
                if result_dict['result']['type'] == 'image':
                    image_data = base64.b64decode(result_dict['result']['base64_data'])
                    image = Image.open(BytesIO(image_data))
                    yield image
            except:
                continue


if "assistant" not in st.session_state:
    load_dotenv()
    python_repl = SessionsPythonREPLTool(
        pool_management_endpoint=os.getenv("POOL_MANAGEMENT_ENDPOINT")
    )
    st.session_state.assistant = LangGraphAssistant(os.getenv("THREAD_ID"), os.getenv("USER_ID"), python_repl)


if "messages" in st.session_state.assistant.get_agent_state().values:
    #print("Summary:")
    #print(st.session_state.assistant.get_agent_state().values["summary"])
    #print()
    #print()
    for message in st.session_state.assistant.get_agent_state().values['messages']:
        message.pretty_print()
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            if len(message.content) > 0:
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        elif isinstance(message, ToolMessage) and message.name == "python_repl":
            try:
                result_dict = message.artifact
                if result_dict['result']['type'] == 'image':
                    with st.chat_message("assistant"):
                        image_data = base64.b64decode(result_dict['result']['base64_data'])
                        image = Image.open(BytesIO(image_data))
                        st.write(image)
            except:
                continue


with st.sidebar:
    file = st.file_uploader("Upload a file")

    if st.button("Set File") and file:
        st.session_state.assistant.set_file(file)


    if st.button("Clear Chat"):
        with MongoClient(os.getenv("MONGODB_URI")) as mongo_client:
            db = mongo_client["checkpointing_db"]
            print(f"Chat record numbers: {db['checkpoints'].count_documents({'thread_id': os.getenv('THREAD_ID')})}")
            db["checkpoints"].delete_many({"thread_id": os.getenv("THREAD_ID")})
            print(f"Chat cleared. Record numbers: {db['checkpoints'].count_documents({'thread_id': os.getenv('THREAD_ID')})}")
            st.rerun()



if prompt := st.chat_input("Text your messages"):
    # 1) Kullanıcı mesajını ekrana bas
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2) Kullanıcı mesajını asistanın stream_answer metoduna gönder
    stream = st.session_state.assistant.stream_answer(prompt)

    # 3) Asistanın cevabını ekrana bas
    with st.chat_message("assistant"):
        st.write_stream(stream_response(stream))
    
    