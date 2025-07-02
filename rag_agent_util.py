import pymupdf4llm
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma, FAISS
import json
import os
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
import re
from langchain_core.documents import Document
from pymupdf import pymupdf




def create_page_chunks_docs(chunks,file_name):
    docs = []
    for chunk in chunks:
        page_no = chunk["metadata"]["page"]
        metadata = {"page_no": page_no, "file_name":file_name}
        docs.append(Document(page_content=chunk["text"], metadata=metadata))

    return docs





def get_md_splits(doc_,file_name):
    chunks = pymupdf4llm.to_markdown(doc_, page_chunks=True, show_progress=True)
    docs = create_page_chunks_docs(chunks,file_name)
    md_split_docs = []
    for doc in docs:
        text = doc.page_content
        metadata = doc.metadata
        md_splitter = MarkdownHeaderTextSplitter([
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
        md_docs = md_splitter.split_text(text)
        for doc in md_docs:
            doc.metadata.update(metadata)
        md_split_docs.extend(md_docs)
    return md_split_docs



def get_splitter(option, embedding, chunk_size=600, chunk_overlap=50):
    if option == "RecursiveCharacterText":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif option == "CharacterText":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif option == "Semantic":
        splitter = SemanticChunker(embeddings=embedding)
    else:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter


def chunk_data(markdown_list, option, embedding, chunk_size=600, chunk_overlap=50):

    splitter = get_splitter(option, embedding, chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap)
    import itertools
    flattened_markdown_list = list(itertools.chain.from_iterable(markdown_list))
    if option == "Sentence":
        recursive_text_splitter = []
        for doc in flattened_markdown_list:
            chunks = splitter.split_text(doc.page_content)
            recursive_text_splitter.extend([Document(page_content=chunk,metadata=doc.metadata) for chunk in chunks])
    else:
        recursive_text_splitter = splitter.split_documents(flattened_markdown_list)
    return recursive_text_splitter




def convert_text_docs(docs):
    from langchain_core.documents import Document
    import uuid
    updated_list = []
    for doc in docs:
        if isinstance(doc, str):
            short_uid = str(uuid.uuid4())
            new_val = Document(page_content=doc, metadata={"id_": short_uid})
            updated_list.append(new_val)
        else:
            updated_list.append(doc)
    return updated_list


@st.cache_data
def extract_text(files_list):
    markdown_list = []

    for file in files_list:
        pdf_data = pymupdf.open(stream=file, filetype="pdf")
        markdown_list_ = get_md_splits(pdf_data,file.name)
        markdown_list.append(markdown_list_)
    return markdown_list


def get_embedding_model(embedd_model,env_var,provider, embedd_model_dir):
    embedding = OpenAIEmbeddings(model=embedd_model,
                                 api_key=env_var.get(
                                     "OPENAI_API_KEY")) if provider == "openai" else HuggingFaceEmbeddings(
        model_name=embedd_model, model_kwargs={"trust_remote_code": True}, cache_folder=embedd_model_dir,
        show_progress=True)
    return embedding


def get_faiss_vectorstore(persistent_dir, embedding, chunks):
    index_file_path = os.path.join(persistent_dir, "index.faiss")

    if os.path.exists(index_file_path):
        vector_store = FAISS.load_local(persistent_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(chunks, embedding)
        vector_store.save_local(persistent_dir)

    return vector_store


def get_chroma_vectorstore(persistent_dir, embedding, chunks):
    if os.path.exists(persistent_dir):
        vector_store = Chroma(persist_directory=persistent_dir, embedding_function=embedding)
    else:
        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persistent_dir)
    return vector_store


def create_vector_store(vs_name, chunks, persistent_dir, embedding):
    vector_store = get_chroma_vectorstore(persistent_dir, embedding,
                                          chunks) if vs_name == "Chroma" else get_faiss_vectorstore(persistent_dir,
                                                                                                    embedding, chunks)
    return vector_store


def get_retriever(docs, vector_store=None, weights=[], k=3):
    keyword_retriever = BM25Retriever.from_documents(docs)
    if not vector_store:
        raise ValueError("Provide vector store")
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retriever = EnsembleRetriever(retrievers=[keyword_retriever, semantic_retriever], weights=weights)
    return retriever


def compute_rerank(user_question, retriever,k=3):
    from flashrank import Ranker
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="~/.cache/flashrank")
    compressor = FlashrankRerank(client=ranker, top_n=k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.invoke(
        user_question
    )
    return compressed_docs


def get_sub_queries(quest, env_var,multi_query=False):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=env_var.get("GROQ_API_KEY", None), temperature=0.2)
    from langchain.prompts import PromptTemplate
    multiquery_prompt = """
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.

Original question: {user_question}

Return the output as a valid Python list of strings. Example format:
[
    "rephrased question 1",
    "rephrased question 2",
    ...
]
if you are unable to split the question then just return the output as 
[
 user_question

]
"""
    sub_query_prompt = """
Perform query decomposition. Given a user question, break it down into distinct sub-questions 
that you would need to answer in order to answer the original question.

Original question: {user_question}

Return the output as a valid Python list of strings. Example format:
[
    "sub-question 1",
    "sub-question 2",
    ...
]

if you are unable to split the question then just return the output as 
[
 user_question

]
"""
    prompt = PromptTemplate.from_template(
        multiquery_prompt if multi_query else sub_query_prompt
    )
    query_analyzer = prompt | llm
    queries = query_analyzer.invoke({"user_question": quest})
    print(queries)
    return eval(queries.content)


def get_sub_query_(user_question, retriver, env_var,sub_query=True, multi_query=False,k=3):
    combined_docs = []
    from flashrank import Ranker
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="~/.cache/flashrank")
    compressor = FlashrankRerank(client=ranker, top_n=k)
    if sub_query:
        sub_queries = get_sub_queries(user_question, env_var,multi_query)
        for query in sub_queries:
            docs = retriver.invoke(query)
            compressed_docs = compressor.compress_documents(docs, query)
            combined_docs.extend(compressed_docs)
        unique_docs = {}
        for doc in combined_docs:
            content = doc.page_content.strip()
            if content not in unique_docs:
                unique_docs[content] = doc
        deduplicated_docs = list(unique_docs.values())
        compressed_docs_new = compressor.compress_documents(deduplicated_docs, user_question)
        return compressed_docs_new
    else:
        return compute_rerank(user_question, retriver,k=k)


def generate_response_model(provider, model_name, user_question, env_var, temperature, retriever,chat_history,sub_query=False,
                            multi_query=False,k=3):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    default_prompt = """
Use the following list of contexts and the chat history to answer the question.

- The given context is in ***markdown*** format — take this into account while interpreting the content.
- You are also provided with previous chat history:
- Use the chat history to understand the user’s intent and clarify references that may not be fully specified in the current question.
- If the context and chat history together do not provide enough information, respond with:
  "The question is irrelevant to the document."
- Focus on **semantic meaning**, not just keyword overlap.
- Pay close attention to **contextual relevance**, coherence, and continuity in the conversation.
- If the context and chat history doesnt  answer the question or irrelevant then dont make up answer just say "context doesnt contain information of the question"

Question: {question}
chat history : {chat_history}
context : {context}
Answer:
        """
    relevant_docs = get_sub_query_(user_question, retriever,env_var ,sub_query=sub_query, multi_query=multi_query,k=k)
    prompt = PromptTemplate.from_template(
        default_prompt
    )
    llm_context = []
    for doc in relevant_docs:
        llm_context.append(doc.page_content if isinstance(doc, Document) else doc)
    llm = ChatOpenAI(model=model_name, api_key=env_var.get("OPENAI_API_KEY", None),
                     temperature=temperature) if provider == "openai" else ChatGroq(model=model_name,
                                                                                    api_key=env_var.get(
                                                                                        "GROQ_API_KEY", None),
                                                                                    temperature=temperature)
    chain = prompt | llm
    response = chain.invoke({"context": llm_context, "question": user_question,"chat_history":chat_history})
    return {"response": response if isinstance(response, str) else response.content, "relevant_docs": relevant_docs}


def remove_think_tags(string):
    import re
    exp = r'<think>(.*?)</think>'
    string = re.sub(exp, "", string, flags=re.DOTALL)
    return string


