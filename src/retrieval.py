"""
Retrieval and Query Processing Module for LCA RAG Application.

This module handles:
- Query embedding and processing
- Top-k retrieval with FlashRank reranking
- Metadata-based filtering
- Multi-source context aggregation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FlashRank for fast reranking
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 10
    similarity_threshold: float = 0.5
    rerank: bool = True
    rerank_top_n: int = 5
    use_flashrank: bool = True  # Use FlashRank for fast reranking
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"  # Fast and accurate
    include_metadata: bool = True
    aggregate_by_source: bool = True


class FlashRankReranker:
    """
    Fast reranker using FlashRank library.
    
    FlashRank provides ultra-fast (<50ms) reranking using
    lightweight cross-encoder models.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize FlashRank reranker.
        
        Args:
            model_name: Model to use for reranking
        """
        self.model_name = model_name
        self._ranker = None
        
        if not FLASHRANK_AVAILABLE:
            logger.warning("FlashRank not available. Install with: pip install flashrank")
    
    def _get_ranker(self) -> Optional["Ranker"]:
        """Lazy load the ranker."""
        if self._ranker is None and FLASHRANK_AVAILABLE:
            try:
                self._ranker = Ranker(model_name=self.model_name)
                logger.info(f"Loaded FlashRank model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load FlashRank: {e}")
        return self._ranker
    
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int = 5
    ) -> List[NodeWithScore]:
        """
        Rerank nodes using FlashRank.
        
        Args:
            query: User query
            nodes: List of nodes to rerank
            top_n: Number of top results to return
            
        Returns:
            Reranked list of nodes
        """
        if not nodes:
            return nodes
        
        ranker = self._get_ranker()
        if ranker is None:
            # Fall back to original order
            return nodes[:top_n]
        
        try:
            # Prepare passages for FlashRank
            passages = [
                {"id": i, "text": node.node.text[:1000]}  # Limit text length
                for i, node in enumerate(nodes)
            ]
            
            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)
            
            # Get reranked results
            results = ranker.rerank(rerank_request)
            
            # Map back to original nodes with updated scores
            reranked = []
            for result in results[:top_n]:
                idx = result["id"]
                node = nodes[idx]
                # Update score with reranker score
                node.score = result["score"]
                reranked.append(node)
            
            logger.debug(f"FlashRank reranked {len(nodes)} -> {len(reranked)} nodes")
            return reranked
            
        except Exception as e:
            logger.warning(f"FlashRank reranking failed: {e}")
            return nodes[:top_n]


@dataclass
class RetrievedContext:
    """Container for retrieved context."""
    text: str
    score: float
    source: str
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryProcessor:
    """
    Processes and enhances queries for better retrieval.
    """
    
    def __init__(self, embed_model: HuggingFaceEmbedding):
        """
        Initialize query processor.
        
        Args:
            embed_model: Embedding model for query encoding
        """
        self.embed_model = embed_model
        
    def process_query(self, query: str) -> QueryBundle:
        """
        Process a query string into a QueryBundle.
        
        Args:
            query: User query string
            
        Returns:
            QueryBundle with embedding
        """
        # Clean and normalize query
        query = query.strip()
        
        # Generate embedding
        embedding = self.embed_model.get_text_embedding(query)
        
        return QueryBundle(
            query_str=query,
            embedding=embedding
        )
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms for better recall.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        # Basic expansion - add domain-specific terms
        expansions = [query]
        
        # LCA-specific expansions
        lca_terms = {
            "impact": ["environmental impact", "GWP", "global warming potential"],
            "carbon": ["CO2", "carbon footprint", "greenhouse gas"],
            "steel": ["iron", "metal", "ferrous"],
            "lifecycle": ["life cycle", "cradle-to-grave", "cradle-to-gate"],
            "assessment": ["analysis", "evaluation", "study"]
        }
        
        query_lower = query.lower()
        for term, alternatives in lca_terms.items():
            if term in query_lower:
                for alt in alternatives:
                    if alt.lower() not in query_lower:
                        expansions.append(query.replace(term, alt))
                        break
        
        return expansions[:3]  # Limit expansions


class ContextRetriever:
    """
    Retrieves relevant context from the vector store.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            index: LlamaIndex VectorStoreIndex
            config: Retrieval configuration
        """
        self.index = index
        self.config = config or RetrievalConfig()
        
        # Create base retriever
        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.config.top_k
        )
        
        # Create similarity filter
        self.similarity_filter = SimilarityPostprocessor(
            similarity_cutoff=self.config.similarity_threshold
        )
        
        # Initialize FlashRank reranker if enabled
        self._flashrank_reranker = None
        if self.config.use_flashrank and FLASHRANK_AVAILABLE:
            self._flashrank_reranker = FlashRankReranker(
                model_name=self.config.flashrank_model
            )
        
    def retrieve(
        self,
        query_bundle: QueryBundle,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query_bundle: Processed query
            filters: Optional metadata filters
            
        Returns:
            List of NodeWithScore objects
        """
        # Retrieve nodes
        nodes = self.retriever.retrieve(query_bundle)
        
        # Apply similarity threshold
        nodes = self.similarity_filter.postprocess_nodes(
            nodes, 
            query_bundle=query_bundle
        )
        
        # Apply metadata filters
        if filters:
            nodes = self._apply_filters(nodes, filters)
        
        # Rerank if enabled
        if self.config.rerank:
            # Use FlashRank if available, otherwise fall back to heuristic
            if self._flashrank_reranker is not None:
                nodes = self._flashrank_reranker.rerank(
                    query_bundle.query_str,
                    nodes,
                    top_n=self.config.rerank_top_n
                )
            else:
                nodes = self._rerank(nodes, query_bundle)
                nodes = nodes[:self.config.rerank_top_n]
        
        return nodes
    
    def _apply_filters(
        self,
        nodes: List[NodeWithScore],
        filters: Dict[str, Any]
    ) -> List[NodeWithScore]:
        """Apply metadata filters to nodes."""
        filtered = []
        for node in nodes:
            match = True
            for key, value in filters.items():
                node_value = node.node.metadata.get(key)
                if isinstance(value, list):
                    if node_value not in value:
                        match = False
                        break
                else:
                    if node_value != value:
                        match = False
                        break
            if match:
                filtered.append(node)
        return filtered
    
    def _rerank(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on relevance.
        
        Simple reranking based on:
        - Original similarity score
        - Content type relevance
        - Source diversity
        """
        if not nodes:
            return nodes
            
        # Score adjustments
        scored_nodes = []
        seen_sources = set()
        
        for node in nodes:
            score = node.score or 0.0
            
            # Boost for text content (usually more informative)
            content_type = node.node.metadata.get("content_type", "text")
            if content_type == "text":
                score *= 1.1
            elif content_type == "table":
                score *= 1.05
                
            # Diversity bonus for new sources
            source = node.node.metadata.get("source", "")
            if source not in seen_sources:
                score *= 1.05
                seen_sources.add(source)
            
            scored_nodes.append((node, score))
        
        # Sort by adjusted score
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return
        result = []
        for node, score in scored_nodes:
            node.score = score
            result.append(node)
            
        return result


class ContextAggregator:
    """
    Aggregates retrieved context from multiple sources.
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize the aggregator.
        
        Args:
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        
    def aggregate(
        self,
        nodes: List[NodeWithScore]
    ) -> Tuple[str, List[RetrievedContext]]:
        """
        Aggregate nodes into a unified context.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Tuple of (formatted context string, list of RetrievedContext)
        """
        if not nodes:
            return "", []
            
        contexts = []
        
        for node in nodes:
            ctx = RetrievedContext(
                text=node.node.text,
                score=node.score or 0.0,
                source=node.node.metadata.get("source", "unknown"),
                content_type=node.node.metadata.get("content_type", "text"),
                metadata=node.node.metadata
            )
            contexts.append(ctx)
        
        # Group by source if enabled
        if self.config.aggregate_by_source:
            formatted = self._format_by_source(contexts)
        else:
            formatted = self._format_flat(contexts)
            
        return formatted, contexts
    
    def _format_by_source(self, contexts: List[RetrievedContext]) -> str:
        """Format contexts grouped by source."""
        by_source = {}
        for ctx in contexts:
            source = ctx.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(ctx)
        
        parts = []
        for source, ctxs in by_source.items():
            source_name = source.split("/")[-1] if "/" in source else source
            source_name = source.split("\\")[-1] if "\\" in source_name else source_name
            
            parts.append(f"### Source: {source_name}")
            for ctx in ctxs:
                parts.append(f"[{ctx.content_type.upper()}] (relevance: {ctx.score:.2f})")
                parts.append(ctx.text)
                parts.append("")
        
        return "\n".join(parts)
    
    def _format_flat(self, contexts: List[RetrievedContext]) -> str:
        """Format contexts in flat list."""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            source_name = ctx.source.split("/")[-1] if "/" in ctx.source else ctx.source
            source_name = source_name.split("\\")[-1] if "\\" in source_name else source_name
            
            parts.append(f"### Context {i}")
            parts.append(f"Source: {source_name}")
            parts.append(f"Type: {ctx.content_type}")
            parts.append(f"Relevance: {ctx.score:.2f}")
            parts.append("")
            parts.append(ctx.text)
            parts.append("")
        
        return "\n".join(parts)


class RAGRetriever:
    """
    Complete retrieval pipeline for RAG.
    
    Combines query processing, retrieval, and context aggregation.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        embed_model: HuggingFaceEmbedding,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            index: Vector store index
            embed_model: Embedding model
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        self.query_processor = QueryProcessor(embed_model)
        self.retriever = ContextRetriever(index, self.config)
        self.aggregator = ContextAggregator(self.config)
        
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        expand_query: bool = False
    ) -> Tuple[str, List[RetrievedContext], List[NodeWithScore]]:
        """
        Full retrieval pipeline.
        
        Args:
            query: User query
            filters: Optional metadata filters
            expand_query: Whether to expand query
            
        Returns:
            Tuple of (formatted context, RetrievedContext list, raw nodes)
        """
        # Process query
        query_bundle = self.query_processor.process_query(query)
        
        # Optionally expand and merge results
        if expand_query:
            expansions = self.query_processor.expand_query(query)
            all_nodes = []
            seen_ids = set()
            
            for exp_query in expansions:
                exp_bundle = self.query_processor.process_query(exp_query)
                nodes = self.retriever.retrieve(exp_bundle, filters)
                
                for node in nodes:
                    if node.node.id_ not in seen_ids:
                        all_nodes.append(node)
                        seen_ids.add(node.node.id_)
            
            # Re-sort merged results
            all_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            nodes = all_nodes[:self.config.top_k]
        else:
            nodes = self.retriever.retrieve(query_bundle, filters)
        
        # Aggregate context
        formatted_context, contexts = self.aggregator.aggregate(nodes)
        
        logger.info(f"Retrieved {len(nodes)} nodes for query: {query[:50]}...")
        
        return formatted_context, contexts, nodes


def main():
    """Test the retrieval pipeline."""
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    # Create embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Create sample nodes
    sample_data = [
        {
            "text": "Life Cycle Assessment (LCA) is a systematic approach to evaluate environmental impacts of products throughout their entire lifecycle from raw material extraction to disposal.",
            "source": "lca_intro.pdf",
            "content_type": "text"
        },
        {
            "text": "Global Warming Potential (GWP) measures greenhouse gas emissions relative to CO2. Steel production typically has a GWP of 2.0-2.5 kg CO2 equivalent per kg.",
            "source": "materials_impact.pdf",
            "content_type": "text"
        },
        {
            "text": "| Material | GWP (kg CO2 eq) | Energy (MJ) |\n|----------|----------------|-------------|\n| Steel | 2.1 | 25 |\n| Aluminum | 8.1 | 155 |",
            "source": "impact_data.xlsx",
            "content_type": "table"
        },
        {
            "text": "ReCiPe methodology provides characterization factors for 18 midpoint and 3 endpoint impact categories including climate change, ozone depletion, and human toxicity.",
            "source": "recipe_guide.pdf",
            "content_type": "text"
        },
        {
            "text": "• LCA Phases\n  • Goal and Scope\n    • Define functional unit\n    • Set system boundaries\n  • Inventory Analysis\n    • Collect data\n  • Impact Assessment\n  • Interpretation",
            "source": "lca_framework.xmind",
            "content_type": "mindmap"
        }
    ]
    
    print("\n=== Creating Sample Index ===")
    nodes = []
    for data in sample_data:
        embedding = embed_model.get_text_embedding(data["text"])
        node = TextNode(
            text=data["text"],
            metadata={
                "source": data["source"],
                "content_type": data["content_type"]
            }
        )
        node.embedding = embedding
        nodes.append(node)
    
    # Create in-memory index
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    print(f"Created index with {len(nodes)} nodes")
    
    # Initialize retriever
    config = RetrievalConfig(
        top_k=5,
        similarity_threshold=0.3,
        rerank=True,
        rerank_top_n=3
    )
    rag_retriever = RAGRetriever(index, embed_model, config)
    
    # Test queries
    test_queries = [
        "What is the environmental impact of steel production?",
        "How does LCA work?",
        "What are the ReCiPe impact categories?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        
        context, retrieved_contexts, nodes = rag_retriever.retrieve(query)
        
        print(f"\nRetrieved {len(retrieved_contexts)} contexts:")
        for i, ctx in enumerate(retrieved_contexts):
            print(f"\n--- Context {i+1} ---")
            print(f"Source: {ctx.source}")
            print(f"Type: {ctx.content_type}")
            print(f"Score: {ctx.score:.4f}")
            print(f"Text: {ctx.text[:100]}...")
        
        print(f"\n--- Formatted Context ---")
        print(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    main()
