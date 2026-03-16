import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate

# 1. Setup Environment
load_dotenv()

# 2. Load and Split the Document
loader = TextLoader("./task-3/ai_intro.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=20,
    separator=" "
)
docs = text_splitter.split_documents(documents)
print(f"Document split into {len(docs)} chunks.")

# 3. Create Vector Store & Embeddings
# AzureOpenAIEmbeddings transforms text into numbers (vectors)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="text-embedding-3-small",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
print("Vector store indexed and retriever ready.")

# 4. Query & Summarize
query = "AI milestones"
retrieved_docs = retriever.invoke(query)

# Combine retrieved text into one string
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Reuse the Summarization Logic from Task 2
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

template = "Summarize the following AI milestones into exactly 3 sentences: {text}"
prompt = PromptTemplate.from_template(template)
chain = prompt | llm

print(f"\n--- Summary of Retrieved Info for '{query}' ---")
response = chain.invoke({"text": context_text})
print(response.content)