import streamlit as st
import requests
import os

FASTAPI_QUERY_URL = "http://localhost:8000/query_embeddings"
FASTAPI_UPLOAD_URL = "http://localhost:8000/generate_embeddings"

st.set_page_config(page_title="RAG Chatbot", page_icon="", layout="centered")

st.title("RAG Chatbot With File Upload")
st.write("Upload documents → generate embeddings → ask questions.")

# -------------------------------
# File Upload Section
# -------------------------------
st.subheader("📤 Upload a Document")

uploaded_file = st.file_uploader(
    "Upload PDF/TXT/DOC file to generate embeddings",
    type=["pdf"]
)

if uploaded_file:
    with st.spinner("Uploading & generating embeddings..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(FASTAPI_UPLOAD_URL, files=files)

            if response.status_code == 200:
                st.success("✅ Embeddings generated successfully!")
            else:
                st.error(f"❌ Error: {response.text}")

        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

st.markdown("---")

# -------------------------------
# Chat Section
# -------------------------------
st.subheader("💬 Chat With Your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask a question...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Send query to FastAPI
    try:
        payload = {"message": prompt}
        response = requests.post(FASTAPI_QUERY_URL, json=payload)

        if response.status_code == 200:
            bot_msg = response.json().get("message", "No response received.")
        else:
            bot_msg = f"⚠️ Error {response.status_code}: {response.text}"

    except Exception as e:
        bot_msg = f"⚠️ Request failed: {e}"

    # Show assistant response
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})
