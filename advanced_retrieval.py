"""
Advanced Retrieval Module with Citation Grounding
==================================================
Implements sophisticated retrieval-augmented generation with
precise source citation and relevance scoring for ManualQ.

Features:
- Multi-stage retrieval ranking
- Relevance confidence scoring
- Citation metadata tracking
- Page-precise answer grounding
"""

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CitedPassage:
    """Represents a retrieved passage with citation information."""
    text: str
    source: str
    page_number: int
    section: str
    relevance_score: float
    distance: float
    chunk_id: int
    
    def format_citation(self) -> str:
        """Format passage with proper citation."""
        return f"{self.text}\n\n[Source: {self.source}, Page {self.page_number}, Section: {self.section}]"


class AdvancedRetriever:
    """
    Multi-stage retriever that combines vector similarity with
    semantic relevance scoring and citation tracking.
    """
    
    def __init__(self, faiss_index=None, chunk_store=None):
        """
        Initialize retriever.
        
        Args:
            faiss_index: FAISS vector index
            chunk_store: Dictionary of chunk ID -> chunk metadata
        """
        self.faiss_index = faiss_index
        self.chunk_store = chunk_store or {}
        self.retrieval_history = []
    
    def retrieve_with_citations(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: float = 0.7
    ) -> List[CitedPassage]:
        """
        Retrieve top-k chunks with citation metadata.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of results to retrieve
            score_threshold: Minimum confidence threshold
            
        Returns:
            List of CitedPassage objects sorted by relevance
        """
        if self.faiss_index is None:
            return []
        
        # Get initial k*2 results for re-ranking
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1),
            min(k * 2, len(self.chunk_store))
        )
        
        passages = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunk_store.get(int(idx))
            if chunk is None:
                continue
            
            # Calculate relevance score (lower FAISS distance = higher relevance)
            relevance_score = max(0, 1 - distance)
            
            if relevance_score < score_threshold:
                continue
            
            passage = CitedPassage(
                text=chunk.get('text', ''),
                source=chunk.get('source', 'Unknown Source'),
                page_number=chunk.get('page', 0),
                section=chunk.get('section', 'Unknown Section'),
                relevance_score=relevance_score,
                distance=float(distance),
                chunk_id=int(idx)
            )
            passages.append(passage)
        
        # Sort by relevance and return top-k
        passages.sort(key=lambda x: x.relevance_score, reverse=True)
        result = passages[:k]
        
        # Track retrieval
        self.retrieval_history.append({
            'query_used': True,
            'results_count': len(result),
            'avg_relevance': np.mean([p.relevance_score for p in result])
        })
        
        return result
    
    def rerank_by_semantic_similarity(
        self,
        passages: List[CitedPassage],
        query_text: str,
        semantic_scorer=None
    ) -> List[CitedPassage]:
        """
        Re-rank passages using semantic similarity.
        
        Args:
            passages: List of retrieved passages
            query_text: Original query text
            semantic_scorer: Function that scores semantic similarity
            
        Returns:
            Re-ranked passages
        """
        if semantic_scorer is None:
            return passages
        
        # Score each passage
        for passage in passages:
            semantic_score = semantic_scorer(query_text, passage.text)
            # Combine FAISS distance score with semantic score
            passage.relevance_score = (passage.relevance_score + semantic_score) / 2
        
        # Re-sort
        passages.sort(key=lambda x: x.relevance_score, reverse=True)
        return passages
    
    def get_grounded_context(
        self,
        passages: List[CitedPassage],
        include_citations: bool = True
    ) -> str:
        """
        Generate context string for LLM with optional citations.
        
        Args:
            passages: List of cited passages
            include_citations: Whether to include citation metadata
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = ["Retrieved context:"]
        
        for i, passage in enumerate(passages, 1):
            if include_citations:
                context_parts.append(
                    f"\n[Source {i}: {passage.source}, Page {passage.page_number}]\n"
                    f"{passage.text}\n"
                    f"Relevance: {passage.relevance_score:.2%}"
                )
            else:
                context_parts.append(f"\n{passage.text}")
        
        return "\n".join(context_parts)
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about retrieval performance."""
        if not self.retrieval_history:
            return {}
        
        return {
            'total_retrievals': len(self.retrieval_history),
            'avg_results_count': np.mean([r['results_count'] for r in self.retrieval_history]),
            'avg_relevance_score': np.mean([r['avg_relevance'] for r in self.retrieval_history]),
        }


class RAGPromptBuilder:
    """
    Builds optimized RAG prompts with context compression
    and citation requirements.
    """
    
    @staticmethod
    def build_rag_prompt(
        query: str,
        context: str,
        system_role: str = "expert assistant"
    ) -> str:
        """
        Build a RAG prompt template.
        
        Args:
            query: User question
            context: Retrieved context with citations
            system_role: Role for the LLM
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a {system_role} answering questions based on the provided context.

Instructions:
1. Answer the question using ONLY information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Always cite the source and page number when referencing information
4. Be concise and direct in your response
5. If multiple sources support your answer, reference all relevant ones

{context}

Question: {query}

Answer (with citations):"""
        
        return prompt
    
    @staticmethod
    def build_citation_aware_prompt(
        query: str,
        passages: List[CitedPassage]
    ) -> Tuple[str, List[Dict]]:
        """
        Build a prompt with explicit citation requirements.
        
        Args:
            query: User question
            passages: Retrieved passages with metadata
            
        Returns:
            Tuple of (prompt_string, citation_metadata_list)
        """
        context_lines = []
        citation_metadata = []
        
        for i, passage in enumerate(passages, 1):
            source_ref = f"[{i}]"
            context_lines.append(f"{source_ref} {passage.text}")
            citation_metadata.append({
                'ref': source_ref,
                'source': passage.source,
                'page': passage.page_number,
                'section': passage.section,
                'relevance': passage.relevance_score
            })
        
        prompt = f"""Answer this question using the provided sources. You MUST cite sources using [number] notation.

Question: {query}

Sources:
{''.join(context_lines)}

Answer (with citations like "According to [1], ..."):"""
        
        return prompt, citation_metadata


def demonstrate_retrieval(sample_passages: List[Dict]) -> None:
    """Demonstrate advanced retrieval capabilities."""
    print("\n" + "="*60)
    print("Advanced Retrieval & Citation System Demo")
    print("="*60 + "\n")
    
    # Mock retriever setup
    retriever = AdvancedRetriever()
    
    # Create sample cited passages
    passages = [
        CitedPassage(
            text="Error E17 indicates a power supply failure.",
            source="Product Manual",
            page_number=45,
            section="Error Codes",
            relevance_score=0.95,
            distance=0.05,
            chunk_id=0
        ),
        CitedPassage(
            text="To resolve power supply issues, check the main power connector.",
            source="Product Manual",
            page_number=120,
            section="Troubleshooting",
            relevance_score=0.87,
            distance=0.13,
            chunk_id=1
        )
    ]
    
    # Build context
    context = retriever.get_grounded_context(passages)
    print("📋 Retrieved Context with Citations:")
    print(context)
    
    # Build RAG prompt
    query = "What does error E17 mean and how do I fix it?"
    rag_prompt = RAGPromptBuilder.build_rag_prompt(query, context)
    print("\n\n🎯 Generated RAG Prompt:")
    print(rag_prompt)
    
    # Citation-aware prompt
    prompt_with_citations, metadata = RAGPromptBuilder.build_citation_aware_prompt(
        query, passages
    )
    print("\n\n📌 Citation-Aware Prompt:")
    print(prompt_with_citations)
    print("\n📑 Citation Metadata:")
    for meta in metadata:
        print(f"   {meta['ref']} - {meta['source']}, Page {meta['page']}, "
              f"Relevance: {meta['relevance']:.1%}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    demonstrate_retrieval([])
