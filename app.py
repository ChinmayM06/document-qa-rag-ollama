# app.py
import streamlit as st
import os
from pathlib import Path
import tempfile
from src.rag_chain import RAGChain
from config.settings import (
    PAGE_TITLE, 
    PAGE_ICON, 
    LAYOUT, 
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR
)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.document_loaded = False
    st.session_state.chat_history = []
    st.session_state.current_file = None

def initialize_rag():
    """Initialize RAG chain."""
    if st.session_state.rag_chain is None:
        with st.spinner("Initializing AI system..."):
            st.session_state.rag_chain = RAGChain()
        st.success("✓ System initialized!")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory."""
    try:
        # Create uploads directory if it doesn't exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def process_document(file_path, filename):
    """Process and load document into RAG chain."""
    try:
        with st.spinner(f"Processing {filename}..."):
            # Clear previous documents if loading a new one
            if st.session_state.document_loaded:
                st.session_state.rag_chain.clear()
            
            # Load document
            num_chunks = st.session_state.rag_chain.load_document(file_path)
            
            # Update state
            st.session_state.document_loaded = True
            st.session_state.current_file = filename
            st.session_state.chat_history = []  # Clear chat history
            
            st.success(f"✓ Document processed successfully! Created {num_chunks} chunks.")
            
            # Show stats
            stats = st.session_state.rag_chain.get_stats()
            st.info(f"📊 Total documents in memory: {stats['total_documents']}")
            
            return True
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return False

def display_chat_message(role, content):
    """Display a chat message."""
    with st.chat_message(role):
        st.markdown(content)

def main():
    # Header
    st.markdown(f'<div class="main-header">{PAGE_ICON} {PAGE_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your documents using AI</div>', unsafe_allow_html=True)
    
    # Initialize RAG chain
    initialize_rag()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=[ext.replace('.', '') for ext in ALLOWED_EXTENSIONS],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)")
            else:
                st.info(f"📄 File: {uploaded_file.name}")
                st.info(f"📦 Size: {file_size_mb:.2f}MB")
                
                # Process button
                if st.button("Process Document", type="primary"):
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        process_document(file_path, uploaded_file.name)
        
        # Show current document
        if st.session_state.document_loaded:
            st.success(f"✓ Active: {st.session_state.current_file}")
            
            # Clear button
            if st.button("Clear Document", type="secondary"):
                st.session_state.rag_chain.clear()
                st.session_state.document_loaded = False
                st.session_state.current_file = None
                st.session_state.chat_history = []
                st.rerun()
        
        # Divider
        st.divider()
        
        # System info
        if st.session_state.rag_chain:
            st.header("ℹ️ System Info")
            stats = st.session_state.rag_chain.get_stats()
            st.metric("Documents Loaded", stats['total_documents'])
            st.metric("Embedding Dimension", stats['embedding_dimension'])
            with st.expander("Model Details"):
                st.write(f"**Embedding Model:** {stats['model_name']}")
                st.write(f"**LLM Model:** {stats['llm_model']}")
        
        # Footer
        st.divider()
        st.caption("Built with Streamlit, FAISS, and Ollama")
    
    # Main chat area
    if not st.session_state.document_loaded:
        # Welcome message
        st.info("👈 Please upload a document from the sidebar to get started!")
        
        # Instructions
        st.markdown("### How to use:")
        st.markdown("""
        1. **Upload** a PDF or TXT document using the sidebar
        2. **Process** the document by clicking the button
        3. **Ask** questions about the document in the chat
        4. Get **AI-powered answers** based on the document content
        """)
        
        # Features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🚀 Fast")
            st.write("Quick document processing and retrieval")
        with col2:
            st.markdown("#### 🎯 Accurate")
            st.write("Context-aware answers from your documents")
        with col3:
            st.markdown("#### 🔒 Local")
            st.write("All processing happens on your machine")
    
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about your document..."):
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": question})
            display_chat_message("user", question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Query RAG chain
                        result = st.session_state.rag_chain.query(
                            question, 
                            return_sources=True
                        )
                        
                        # Display answer
                        st.markdown(result['answer'])
                        
                        # Display sources in expander
                        if result.get('sources'):
                            with st.expander(f"📚 View {len(result['sources'])} source(s)"):
                                for i, source in enumerate(result['sources'], 1):
                                    st.markdown(f"**Source {i}** (Relevance: {1/(1+source['score']):.2%})")
                                    st.markdown(f'<div class="source-box">{source["content"]}</div>', 
                                              unsafe_allow_html=True)
                                    if source.get('metadata'):
                                        st.caption(f"From: {source['metadata'].get('source', 'Unknown')}")
                                    st.divider()
                        
                        # Add assistant message to chat
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": result['answer']
                        })
                        
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

if __name__ == "__main__":
    main()