from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Qdrant
import pandas as pd
from dotenv import load_dotenv
import os
import glob
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
# Load the .env file
load_dotenv()
# Access the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please add it to the .env file.")
def add_human_prompt(content):
    messages.append(HumanMessage(content=content))
def get_response(prompt):
    # augment prompt
    augmented_prompt = custom_prompt(qdrant_cusail, prompt)
    # process prompt
    add_human_prompt(augmented_prompt)
    # send to GPT
    res = chat.invoke(messages)
    # return response
    return res.content

def load_data():
    
    directory_path = 'data/raw'
    extensions = ["md", "tex"]

    # Collect all matching file paths
    file_paths = []
    for ext in extensions:
        file_paths.extend(glob.glob(f"{directory_path}/*.{ext}"))

    all_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            text = file.read()
        
        # Chunking (currently rudimentary)
        def split_text_into_chunks(text, chunk_size=500):
            words = text.split()
            chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            return chunks
        
        chunks = split_text_into_chunks(text)
        
        for chunk in chunks:
            all_data.append({'chunk': chunk, 'source': file_path})
    
    df = pd.DataFrame(all_data)
    
    #Load the DataFrame as documents
    loader = DataFrameLoader(df, page_content_column="chunk")
    documents = loader.load()
    
    return documents

def qdrant_client():
    cusail_documents = load_data()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    url = 'http://localhost:6333'
    return Qdrant.from_documents(
        documents=cusail_documents,
        embedding=embeddings,
        url=url,
        collection_name="cusail_chatbot",
        )

def custom_prompt(client, query: str):
    results = client.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augment_prompt = f"""Using the contexts below, answer the query:
    Contexts:
    {source_knowledge}
    Query: {query}"""
    return augment_prompt
qdrant_cusail = qdrant_client()
chat = ChatOpenAI(
    model='gpt-3.5-turbo'
)
messages = []


#test querying
question = "Who are the authors of the 'Sailing Algorithm' section of the fall 2023 report?"
answer = get_response(question)
print(answer)