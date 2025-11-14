from fastapi import FastAPI, File, UploadFile, HTTPException
from app.gen_embeddings import GenEmbeddings
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from app.schemas import ChatInput, ChatOutput
from app.rag_agent import rag_agent


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "App is Running Successfully"}

@app.post("/generate_embeddings")
async def generate_embeddings(file: UploadFile):
    # Save the uploaded file to a temporary location /Users/ed/Desktop/BackupScripts/Examples/Agentic RAG/temp
    print("File uploaded successfully")
    temp_folder = "./temp"
    file_path = f"{temp_folder}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    # Create an instance of GenEmbeddings
    gen_embeddings = GenEmbeddings(file_path)
    # Call the store_embeddings method
    gen_embeddings.store_embeddings()
    return

@app.post("/query_embeddings")
async def query_embeddings(input: ChatInput) -> ChatOutput:
    try:
        input_msg = input.message
        print(input_msg)

        user_message = HumanMessage(content=input.message)
        print(user_message)
        input = {"messages": [{"role": "user", "content": input.message}]}
        response = rag_agent.invoke(input)
        # Extract AI final response
        ai_msg = response["messages"][-1]   # Last message is AI
        content = ai_msg.content            # The actual answer text

        return ChatOutput(message=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
