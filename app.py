"""
ManualQ - Streamlit Chat Interface
==================================
Interactive web interface for querying product manuals with GenAI RAG.
"""

import streamlit as st
import os
from pathlib import Path
import json
from rag_pipeline import ManualQPipeline

# Page configuration
st.set_page_config(
    page_title="ManualQ - Chat with Manuals",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'manual_loaded' not in st.session_state:
    st.session_state.manual_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("⚙️ ManualQ Settings")
    st.divider()
    
    # Manual upload
    st.subheader("📄 Upload Manual")
    uploaded_file = st.file_uploader(
        "Choose a PDF manual",
        type=["pdf"],
        help="Upload a product manual (PDF format)"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_dir = Path("./uploaded_manuals")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Process Manual", use_container_width=True):
            with st.spinner("Processing manual... This may take a moment."):
                try:
                    st.session_state.pipeline = ManualQPipeline(
                        embedding_model="all-MiniLM-L6-v2"
                    )
                    result = st.session_state.pipeline.process_manual(str(temp_path))
                    st.session_state.manual_loaded = True
                    st.success(f"✅ Manual processed! Created {result['chunks']} semantic chunks.")
                    
                    # Display statistics
                    with st.expander("📊 Processing Statistics"):
                        stats = result['stats']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pages", stats['total_pages'])
                        with col2:
                            st.metric("Chunks Created", stats['chunks'])
                        with col3:
                            st.metric("Compression", f"{stats['compression_ratio']:.1f}%")
                        
                        with st.expander("Detailed Stats"):
                            st.json(stats)
                except Exception as e:
                    st.error(f"❌ Error processing manual: {str(e)}")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Query Settings")
    retrieval_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More chunks = more context but slower retrieval"
    )
    
    use_llm = st.checkbox(
        "Use LLM for answer generation",
        value=False,
        help="Requires OpenAI API key in environment"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.title("📖 ManualQ")
st.subtitle("Chat with your product manuals using GenAI RAG")

if not st.session_state.manual_loaded:
    st.info(
        "👈 **Get started:** Upload a product manual in the sidebar to begin. "
        "ManualQ will compress, chunk, and index your manual for instant Q&A."
    )
    
    # Show features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🗜️ Context Compression**
        - Removes headers, footers, boilerplate
        - 40-60% token reduction
        """)
    with col2:
        st.markdown("""
        **✂️ Semantic Chunking**
        - Meaningful section splitting
        - Better retrieval precision
        """)
    with col3:
        st.markdown("""
        **📑 Page Citations**
        - Exact page references
        - Verifiable answers
        """)

else:
    # Display pipeline stats
    st.success("✅ Manual loaded and indexed. Ready to answer questions!")
    
    if st.session_state.pipeline:
        with st.expander("📊 Pipeline Statistics"):
            stats = st.session_state.pipeline.get_pipeline_stats()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", stats.get('total_chunks', 0))
            with col2:
                st.metric("Embedding Dim", stats.get('embedding_dimension', 0))
            with col3:
                st.metric("Compression", f"{stats.get('compression_ratio', 0):.1f}%")
            with col4:
                st.metric("Index Size", stats.get('index_size', 0))
    
    st.divider()
    
    # Chat interface
    st.subheader("💬 Ask a Question")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "passages" in message:
                with st.expander(f"📚 Sources ({len(message['passages'])} found)"):
                    for i, p in enumerate(message["passages"], 1):
                        st.markdown(f"**Source {i}:** {p['source']} (Page {p['page']})")
                        st.caption(f"Relevance: {p['relevance']:.1%}")
                        st.markdown(f"> {p['text'][:200]}...")
    
    # Query input
    user_question = st.chat_input(
        "Ask a question about the manual...",
        placeholder="e.g., 'How do I troubleshoot error code E17?'"
    )
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Get assistant response
        with st.spinner("🔍 Searching manual..."):
            try:
                result = st.session_state.pipeline.query(
                    user_question,
                    k=retrieval_k,
                    use_llm=use_llm
                )
                
                if result['success']:
                    # Format response
                    if 'answer' in result:
                        response_text = result['answer']
                    else:
                        response_text = f"""
**Retrieved Information:**

{result['context']}

**Note:** To generate a full answer, use LLM mode in settings (requires OpenAI API key).
"""
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "passages": result['passages']
                    })
                    
                    # Display response
                    with st.chat_message("assistant"):
                        st.write(response_text)
                        with st.expander(f"📚 Sources ({len(result['passages'])} found)"):
                            for i, p in enumerate(result['passages'], 1):
                                st.markdown(f"**Source {i}:** {p['source']} (Page {p['page']})")
                                st.caption(f"Relevance: {p['relevance']:.1%} | Section: {p['section']}")
                                st.markdown(f"> {p['text'][:300]}...")
                else:
                    st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure you have all dependencies installed: `pip install -r requirements.txt`")

# Footer
st.divider()
st.markdown("""
---
**ManualQ** - GenAI RAG System for Product Manuals  
Built with Streamlit • FAISS • Sentence Transformers • OpenAI  
[GitHub](https://github.com/advaiithh/-Product-Manual-Assistant)
""")
