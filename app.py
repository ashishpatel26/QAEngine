import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pprint import pprint
from dotenv import load_dotenv
# 0. Load the environment variables
load_dotenv(dotenv_path=".env")

###### Loading Modules ######
# 1.Read the PDF file
loader = PyPDFLoader(r"docs\2023 Tesla Q1 Earnings Report.pdf")
# 2. Load the PDF file
docs = loader.load()

##### Splitting the Document ######
# Formatting docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# 3. Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 4. Split the document into chunks
splits = text_splitter.split_documents(docs)

##### Define the Embeddings ######
# 5. Initialize the embeddings model
embeddings_model = OpenAIEmbeddings(disallowed_special=())

##### Define the Vector Store ######
# 6. Initialize the vector store
vector_store = Chroma.from_documents(splits, embeddings_model, persist_directory="./chroma_db")

##### Vector Store Retrieval ######
load_vector_store = Chroma(persist_directory = "./chroma_db", embedding_function=embeddings_model)
retriever = load_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

##### Query the Vector Store ######
# result = retriever.invoke("What is the Profitability of Tesla?")

##### Initialize the LLM Model - Openai ######
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

###### Prompt template #####
template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


###### Define the pipeline ######
ragchain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

que = input("Enter your question:")
print(ragchain.invoke(que))






