"""
RAG Pipeline Orchestrator for ManualQ
====================================
Orchestrates the complete retrieval-augmented generation pipeline.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from pdf_processor import PDFProcessor, TextCleaner
from context_compression import ContextCompressor, SemanticChunker
from vector_store import VectorStore, EmbeddingModel
from advanced_retrieval import AdvancedRetriever, CitedPassage, RAGPromptBuilder


class ManualQPipeline:
    """Complete RAG pipeline for ManualQ."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: Name of embedding model to use
        """
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.compressor = ContextCompressor()
        self.chunker = SemanticChunker(max_chunk_size=500, overlap=50)
        self.embedding_model = EmbeddingModel(embedding_model)
        self.vector_store = VectorStore(
            embedding_dim=self.embedding_model.get_embedding_dim()
        )
        self.retriever = AdvancedRetriever(
            faiss_index=self.vector_store.index,
            chunk_store=self.vector_store.chunks
        )
        self.pipeline_stats = {}
    
    def process_manual(self, pdf_path: str) -> Dict:
        """
        Process a PDF manual through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with processing results and statistics
        """
        print(f"🚀 Starting ManualQ pipeline for: {pdf_path}")
        
        # Step 1: Extract PDF
        print("📄 Step 1: Extracting PDF...")
        extraction_result = self.pdf_processor.extract_from_pdf(pdf_path)
        raw_text = extraction_result['text']
        metadata = extraction_result['metadata']
        print(f"   ✓ Extracted {metadata['total_pages']} pages")
        
        # Step 2: Clean text
        print("🧹 Step 2: Cleaning text...")
        cleaned_text = self.text_cleaner.remove_headers_footers(raw_text)
        cleaned_text = self.text_cleaner.normalize_text(cleaned_text)
        print(f"   ✓ Text cleaned")
        
        # Step 3: Compress context
        print("🗜️  Step 3: Compressing context...")
        compression_result = self.compressor.compress(cleaned_text)
        compressed_text = compression_result['compressed_text']
        compression_ratio = compression_result['compression_ratio']
        print(f"   ✓ Compression: {compression_ratio:.1f}% reduction")
        
        # Step 4: Semantic chunking
        print("✂️  Step 4: Semantic chunking...")
        chunks = self.chunker.chunk(
            compressed_text,
            metadata={'source': metadata['filename'], 'page': 1}
        )
        print(f"   ✓ Created {len(chunks)} semantic chunks")
        
        # Step 5: Create embeddings
        print("🧠 Step 5: Creating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(chunk_texts)
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        
        # Step 6: Index vectors
        print("📑 Step 6: Indexing vectors...")
        self.vector_store.add_chunks(chunks, embeddings)
        print(f"   ✓ Indexed in FAISS")
        
        # Update stats
        self.pipeline_stats = {
            'pdf_file': metadata['filename'],
            'total_pages': metadata['total_pages'],
            'raw_tokens': compression_result['original_tokens'],
            'compressed_tokens': compression_result['compressed_tokens'],
            'compression_ratio': compression_ratio,
            'chunks': len(chunks),
            'embedding_model': self.embedding_model.model_name,
        }
        
        print(f"\n✅ Pipeline complete!")
        return {
            'success': True,
            'chunks': len(chunks),
            'stats': self.pipeline_stats,
            'metadata': metadata
        }
    
    def query(self, question: str, k: int = 5, use_llm: bool = False) -> Dict:
        """
        Query the processed manual.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            use_llm: Whether to generate answer with LLM (requires API key)
            
        Returns:
            Dictionary with retrieved passages and answer
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(question)
        
        # Retrieve similar chunks
        passages = self.retriever.retrieve_with_citations(query_embedding, k=k)
        
        if not passages:
            return {
                'success': False,
                'error': 'No relevant information found',
                'passages': []
            }
        
        # Build context
        context = self.retriever.get_grounded_context(passages, include_citations=True)
        
        # Build RAG prompt
        rag_prompt = RAGPromptBuilder.build_rag_prompt(question, context)
        
        result = {
            'success': True,
            'question': question,
            'passages': [
                {
                    'text': p.text,
                    'source': p.source,
                    'page': p.page_number,
                    'section': p.section,
                    'relevance': float(p.relevance_score)
                }
                for p in passages
            ],
            'context': context,
            'rag_prompt': rag_prompt,
        }
        
        if use_llm:
            try:
                import openai
                # Note: Requires OPENAI_API_KEY environment variable
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant answering questions about product manuals."},
                        {"role": "user", "content": rag_prompt}
                    ],
                    temperature=0.7,
                )
                result['answer'] = response.choices[0].message.content
            except Exception as e:
                result['answer_error'] = str(e)
        
        return result
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            **self.pipeline_stats,
            **self.vector_store.get_stats(),
            'retrieval_stats': self.retriever.get_retrieval_stats()
        }


if __name__ == "__main__":
    print("ManualQ RAG Pipeline initialized and ready to process documents.")
