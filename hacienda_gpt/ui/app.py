import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from hacienda_gpt.llm.chain import create_openai_chain
from hacienda_gpt.utils import MissingOpenAIAPIKeyError, configure_logging, get_openai_api_key

# Custom image for the app icon and the assistant's avatar
bot_logo = "https://sede.agenciatributaria.gob.es/static_files/Sede/Tema/Agencia_tributaria/Memorias/2018/Imagenes/Introduccion.jpg"


@st.cache_resource
def load_chain():
    try:
        openai_api_key = get_openai_api_key()
    except MissingOpenAIAPIKeyError:
        st.error("Falta OPENAI_API_KEY en el entorno. Configúrala para usar HaciendaGPT.")
        st.stop()
    return create_openai_chain(openai_api_key=openai_api_key)


def _build_chat_history(messages: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
    history: list[HumanMessage | AIMessage] = []
    for message in messages:
        if message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))
        elif message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
    return history


def main():
    configure_logging()
    st.set_page_config(page_title="HaciendaGPT", page_icon=":bank:", layout="centered")
    st.title("HaciendaGPT")

    chain = load_chain()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "¡Hola! ¿Cómo puedo ayudarte con tus preguntas relacionadas con la Agencia Tributaria?",
            }
        ]

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=bot_logo):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if query := st.chat_input("Preguntáme lo que quieras"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar=bot_logo):
            message_placeholder = st.empty()
            history = _build_chat_history(st.session_state.messages[:-1])
            result = chain.invoke({"input": query, "chat_history": history})
            response = result["answer"]
            full_response = ""

            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
