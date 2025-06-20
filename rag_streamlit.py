import streamlit as st
from rag_agent_util import extract_text, chunk_data, create_vector_store, generate_response_model, \
    get_retriever, get_embedding_model, remove_think_tags

st.title("RAG AGENT")
st.header("Ask Questions get answers!")
if "messages" not in st.session_state:
    st.session_state["messages"] = []



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
files = st.file_uploader("Upload the files here", type=["pdf"], accept_multiple_files=True)
with st.sidebar:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    openapi_key = st.sidebar.text_input("Enter openai api key", type="password")
    groq_api_key = st.sidebar.text_input("Enter groq api key", type="password")
if files:
    unique_data = set()
    unique_files = []
    for file in files:
        if file.name not in unique_data:
            unique_data.add(file.name)
            unique_files.append(file)
    markdown_text = extract_text(unique_files)
    st.code(markdown_text[0:200], "markdown")
    updated_chunks, vector_store = None, None
    with st.form("RAG parameters"):
        st.subheader("Chunking Techniques")
        chunking_option = st.selectbox(
            "Select the chunking technique?",
            ["-- Select an option --", "CharacterText", "RecursiveCharacterText", "Semantic", "Sentence"]
        )
        chunk_size = st.number_input("select the chunk size", min_value=0, max_value=1000, value=500, step=1)
        chunk_overlap = st.number_input("select the chunk overlap size", min_value=0, max_value=1000, value=100, step=1)
        st.subheader("Embedding type")
        provider = st.selectbox("select the embedding model provider",
                                ["-- Select an option --", "openai", "huggingface"])
        embedding_model = st.text_input("Provide the embedding model name")
        st.subheader("Vector Store")
        vector_store_ = st.selectbox("Select the vector store",
                                     ["-- Select an option --", "Chroma", "Faiss"])
        persistent_directory = st.text_input("Enter the persistent directory")
        embedd_model_dir = st.text_input("Enter the embedding model directory")
        st.subheader("Retriever Types")
        st.write("Select the keyword weight:")
        keyword_weight = st.slider("Keyword weight", 0.0, 1.0, 0.2, step=0.1)
        st.write("Semantic weight will be remaining weight will be adjusted to semantic")
        top_k = st.slider("k value", 1, 20, 3, step=1)
        question = st.text_input("Enter the question here")
        llm_provider = st.selectbox("Select the llm provider",
                                    ["-- Select an option --", "openai", "groq"])
        llm_model = st.text_input("Provide the llm model name")
        sub_query = st.checkbox("sub query technique")
        multi_query = st.checkbox("multi query technique")
        submitted = st.form_submit_button("Submit")
        chunked_data = []
        if submitted:
            semantic_weight = 1.0 - keyword_weight
            with st.spinner("Creating Chunking data...."):
                embedding = get_embedding_model(embedding_model, provider, embedd_model_dir)
                updated_chunks = chunk_data(markdown_text, chunking_option, embedding, chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap)
            env_var = {
                "OPENAI_API_KEY": openapi_key,
                "GROQ_API_KEY": groq_api_key
            }
            with st.spinner("Creating/Indexing from vector store"):
                vector_store = create_vector_store(vector_store_, updated_chunks, persistent_directory, embedding)
    if updated_chunks and vector_store:
        retriever = get_retriever(updated_chunks, vector_store, weights=[keyword_weight, semantic_weight], k=4)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Generating model response"):
            st.write(multi_query)
            response = generate_response_model(llm_provider, llm_model, question, env_var, temperature, retriever,
                                               st.session_state.messages, sub_query=sub_query,
                                               multi_query=multi_query, k=top_k)
            st.subheader("Response:")
            resp = remove_think_tags(response["response"])
            st.session_state.messages.append({"role": "ai", "content": resp})
            with st.chat_message("ai"):
                st.markdown(resp)
                relevant_docs = response["relevant_docs"]
                st.subheader("Metadata")
                st.write([doc.metadata for doc in relevant_docs])
                st.subheader("context")
                st.write([doc.page_content for doc in relevant_docs])
