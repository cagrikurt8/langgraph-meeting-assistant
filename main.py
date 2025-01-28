from LangGraphAssistant import LangGraphAssistant
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
import os


st.set_page_config(
    page_title="LangGraph Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)


def stream_response(stream):
    for token in stream:
        if isinstance(token[0], AIMessage):
            if len(token[0].content) > 0:
                yield token[0].content


if "assistant" not in st.session_state:
    load_dotenv()
    st.session_state.assistant = LangGraphAssistant("12345", os.getenv("USER_ID"))

if "messages" in st.session_state.assistant.get_agent_state().values:
    #print(st.session_state.assistant.get_agent_state())
    for message in st.session_state.assistant.get_agent_state().values['messages']:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            if len(message.content) > 0:
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        elif isinstance(message, ToolMessage) and message.name == "python_repl":
            result_dict = json.loads(message.content)
            if result_dict['result']['type'] == 'image':
                with st.chat_message("assistant"):
                    image_data = base64.b64decode(result_dict['result']['base64_data'])
                    image = Image.open(BytesIO(image_data))
                    st.write(image)


if prompt := st.chat_input("Text your messages"):
    # 1) Kullanıcı mesajını ekrana bas
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2) Kullanıcı mesajını asistanın stream_answer metoduna gönder
    stream = st.session_state.assistant.stream_answer(prompt)

    # 3) Asistanın cevabını ekrana bas
    with st.chat_message("assistant"):
        st.write_stream(stream_response(stream))
    
    