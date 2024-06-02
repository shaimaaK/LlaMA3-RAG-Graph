import streamlit as st
from LlaMA3RAGraph import init_model

st.set_page_config(page_title="Derivian Buddy", layout='wide',page_icon='👨‍💻')
st.title("DerivianBuddy")
st.text("Get to know more about deriv from your buddy! ask away")
st.markdown("""---""")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.app = init_model()
    



for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt:=st.chat_input("write your message here"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role":"user", "content":prompt})

    #store the convo
    convo= f"user:{prompt} \n\n"
    for msg in st.session_state.chat_history:
        if msg['role'] =="user":
            convo=convo+f" {msg['content']} \n\n"
        else:
            convo=convo+f" {msg['content']} \n\n"
    with st.spinner("Generate LlaMA3 RAG Graph Response"):
        for output in st.session_state.app.stream({"question":convo}):
            for key, value in output.items():
                pass
        response=value["generation"]
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role":"assistant", "content":response})

