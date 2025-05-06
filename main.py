import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from joke_agent import graph, State

load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot")

with st.sidebar:
    st.write("## Langgraph Graph diagram")
    st.write("This is a graph of the chatbot's knowledge and capabilities.")
    graph_bytes = graph.get_graph().draw_mermaid_png()
    st.image(graph_bytes, caption="Chatbot Graph")
    


# Display the chat history.
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Handle user input.
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            state = State(messages=[user_message])
            

            # invoke the agent, streaming tokens from any llm calls directly
            for chunk, metadata in graph.stream(state, config={"configurable": {"thread_id": "thread"}}, stream_mode="messages"):
                if isinstance(chunk, AIMessage):
                    full_response = full_response + str(chunk.content)
                    message_placeholder.markdown(full_response + "▌")

                elif isinstance(chunk, ToolMessage):
                    full_response = full_response + f"🛠️ Used tool to get: {chunk.content}\n\n"
                    message_placeholder.markdown(full_response + "▌")

            # Once streaming is complete, display the final message without the cursor
            message_placeholder.markdown(full_response)

            # Add the complete message to session state
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}") 