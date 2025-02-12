import os
import streamlit as st

# ------------------------------------------------
# Custom CSS Styling for an aesthetic dark UI
# ------------------------------------------------
st.markdown(
    """
    <style>
    /* General App styling */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    /* File uploader styling */
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    /* Headings styling */
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# App Title and Description
# ------------------------------------------------
st.title("📘DeepSeek QA with CAMEL-AI")
st.markdown("### Your Intelligent Document Assistant\nUpload a document and ask questions to get insights based on its content.")

# ------------------------------------------------
# Sidebar: API Key Configuration
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.text_input("Enter your OpenAI API Key", type="password")
    deepseek_key = st.text_input("Enter your DeepSeek API Key", type="password")
    if openai_key and deepseek_key:
        st.success("API Keys set!")
    st.markdown("---")
    st.markdown("Built with [CAMEL-AI](https://github.com/camel-ai/camel)")

if not (openai_key and deepseek_key):
    st.warning("Please enter both API keys in the sidebar to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["DEEPSEEK_API_KEY"] = deepseek_key

# ------------------------------------------------
# File Upload Section
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
file_path = None
if uploaded_file:
    os.makedirs("uploaded_files", exist_ok=True)
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded and saved.")

# ------------------------------------------------
# Import CAMEL Components for DeepSeek QA
# ------------------------------------------------
from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.retrievers import AutoRetriever
from camel.types import StorageType

embedding_instance = OpenAIEmbedding(model_type=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE)

def single_agent(query: str) -> str:
    if not file_path:
        return "No document uploaded. Please upload a file."
    
    assistant_sys_msg = (
        "You are a helpful assistant to answer questions. "
        "I will give you the Original Query and Retrieved Context, "
        "answer the Original Query based on the Retrieved Context, "
        "if you can't answer the question just say I don't know."
    )
    
    auto_retriever = AutoRetriever(
        vector_storage_local_path="local_data2/",
        storage_type=StorageType.QDRANT,
        embedding_model=embedding_instance
    )

    retrieved_info = auto_retriever.run_vector_retriever(
        query=query,
        contents=[file_path],
        top_k=1,
        return_detailed_info=False,
        similarity_threshold=0.5
    )
    
    if not retrieved_info:
        return "No relevant information found in the document."
    
    deepseek_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_REASONER,
    )
    
    user_msg = str(retrieved_info)
    agent = ChatAgent(
        system_message=assistant_sys_msg,
        model=deepseek_model
    )

    assistant_response = agent.step(user_msg)
    return assistant_response.msg.content if assistant_response else "No response from the agent."

user_query = st.text_input("Enter your question about the document...")
if user_query:
    with st.spinner("Processing your query..."):
        answer = single_agent(user_query)
    st.markdown(f"**Answer:** {answer}")
