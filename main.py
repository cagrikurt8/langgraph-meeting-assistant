from LangGraphAssistant import LangGraphAssistant
import asyncio
import streamlit as st
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(
    page_title="LangGraph Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)


async def async_get_response(stream):
    async for event in stream:
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node', '') == "assistant":
            data = event["data"]
            yield data["chunk"].content


def sync_get_response(stream: AsyncGenerator):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                yield loop.run_until_complete(anext(stream))
            except StopAsyncIteration:
                break
    finally:
        loop.close()


if "assistant" not in st.session_state:
    st.session_state.assistant = LangGraphAssistant("12345")

if "messages" in st.session_state.assistant.get_agent_state().values:
    for message in st.session_state.assistant.get_agent_state().values['messages']:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            if len(message.content) > 0:
                with st.chat_message("assistant"):
                    st.markdown(message.content)


if prompt := st.chat_input("Text your messages"):
    # 1) Kullanıcı mesajını ekrana bas
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2) Kullanıcı mesajını asistanın stream_answer metoduna gönder
    stream = st.session_state.assistant.stream_answer(prompt)

    # 3) Asistanın cevabını ekrana bas
    with st.chat_message("assistant"):
        st.write_stream(sync_get_response(async_get_response(stream)))
    
    