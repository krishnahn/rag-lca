"""
Main RAG Pipeline for LCA Documents.

This is the main entry point for the LCA RAG application.
It orchestrates all components:
- Data ingestion and parsing
- Chunking and embedding
- Vector database indexing
- Retrieval and query processing
- LLM generation
- Evaluation

Usage:
    python -m src.pipeline --help
    python -m src.pipeline ingest --data-dir ./LCA
    python -m src.pipeline query "What is the GWP of steel?"
    python -m src.pipeline interactive
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    
    # Paths
    data_dir: Path = Path("./LCA")
    vector_db_path: Path = Path("./data/qdrant")
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda"
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 128
    
    # Vector DB settings
    collection_name: str = "lca_documents"
    
    # LLM settings
    llm_model: str = "google/gemini-2.0-flash-001"
    llm_temperature: float = 0.1
    
    # Retrieval settings
    top_k: int = 10
    rerank_top_n: int = 5
    similarity_threshold: float = 0.5
    
    # Evaluation settings
    enable_evaluation: bool = True
    faithfulness_threshold: float = 0.8


class LCARagPipeline:
    """
    Complete RAG Pipeline for LCA Documents.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Component instances (lazy loaded)
        self._ingestion_pipeline = None
        self._chunking_pipeline = None
        self._indexer = None
        self._index = None
        self._embed_model = None
        self._retriever = None
        self._llm = None
        self._generator = None
        self._evaluator = None
        
        # Ensure directories exist
        self.config.vector_db_path.mkdir(parents=True, exist_ok=True)
        
    # =========================================================================
    # Component Initialization (Lazy Loading)
    # =========================================================================
    
    def _get_embed_model(self):
        """Get or create embedding model."""
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            self._embed_model = HuggingFaceEmbedding(
                model_name=self.config.embedding_model,
                device=self.config.embedding_device
            )
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        return self._embed_model
    
    def _get_ingestion_pipeline(self):
        """Get or create ingestion pipeline."""
        if self._ingestion_pipeline is None:
            from src.ingestion import DataIngestionPipeline
            self._ingestion_pipeline = DataIngestionPipeline()
        return self._ingestion_pipeline
    
    def _get_chunking_pipeline(self):
        """Get or create chunking pipeline."""
        if self._chunking_pipeline is None:
            from src.chunking import ChunkingEmbeddingPipeline, ChunkingConfig
            
            chunk_config = ChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            self._chunking_pipeline = ChunkingEmbeddingPipeline(
                chunking_config=chunk_config,
                model_name=self.config.embedding_model,
                device=self.config.embedding_device
            )
        return self._chunking_pipeline
    
    def _get_indexer(self):
        """Get or create vector DB indexer."""
        if self._indexer is None:
            from src.indexing import QdrantIndexer, VectorDBConfig
            
            db_config = VectorDBConfig(
                collection_name=self.config.collection_name,
                use_local=True,
                local_path=str(self.config.vector_db_path),
                embedding_dim=384  # MiniLM dimension
            )
            
            self._indexer = QdrantIndexer(db_config)
        return self._indexer
    
    def _get_index(self):
        """Get or create vector store index."""
        if self._index is None:
            from src.indexing import VectorIndexBuilder
            
            indexer = self._get_indexer()
            embed_model = self._get_embed_model()
            
            builder = VectorIndexBuilder(indexer, embed_model)
            self._index = builder.load_index()
            
            if self._index is None:
                logger.warning("No existing index found. Please run ingestion first.")
        
        return self._index
    
    def _get_retriever(self):
        """Get or create retriever."""
        if self._retriever is None:
            from src.retrieval import RAGRetriever, RetrievalConfig
            
            index = self._get_index()
            if index is None:
                raise RuntimeError("Index not available. Please run ingestion first.")
            
            config = RetrievalConfig(
                top_k=self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
                rerank=True,
                rerank_top_n=self.config.rerank_top_n
            )
            
            self._retriever = RAGRetriever(
                index=index,
                embed_model=self._get_embed_model(),
                config=config
            )
        return self._retriever
    
    def _get_llm(self):
        """Get or create LLM."""
        if self._llm is None:
            from src.generation import OpenRouterLLM, LLMConfig

            llm_config = LLMConfig(
                model_name=self.config.llm_model,
                temperature=self.config.llm_temperature
            )

            self._llm = OpenRouterLLM(llm_config)
        return self._llm
    
    def _get_generator(self):
        """Get or create generator."""
        if self._generator is None:
            from src.generation import RAGGenerator
            
            self._generator = RAGGenerator(self._get_llm())
        return self._generator
    
    def _get_evaluator(self):
        """Get or create evaluator."""
        if self._evaluator is None:
            from src.evaluation import RAGEvaluator, EvaluationConfig
            
            eval_config = EvaluationConfig(
                faithfulness_threshold=self.config.faithfulness_threshold
            )
            
            self._evaluator = RAGEvaluator(eval_config, self._get_llm())
        return self._evaluator
    
    # =========================================================================
    # Pipeline Operations
    # =========================================================================
    
    def ingest(self, data_dir: Optional[Path] = None, recreate: bool = False) -> Dict[str, Any]:
        """
        Ingest documents from a directory.
        
        Args:
            data_dir: Directory containing documents
            recreate: Whether to recreate the index
            
        Returns:
            Ingestion statistics
        """
        data_dir = data_dir or self.config.data_dir
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Starting ingestion from: {data_dir}")
        
        # Step 1: Ingest documents
        print("\n" + "="*60)
        print("PHASE 1: Document Ingestion")
        print("="*60)
        
        ingestion = self._get_ingestion_pipeline()
        documents = ingestion.ingest_directory(data_dir)
        
        print(f"✓ Ingested {len(documents)} documents")
        
        # Step 2: Chunk and embed
        print("\n" + "="*60)
        print("PHASE 2: Chunking and Embedding")
        print("="*60)
        
        chunking = self._get_chunking_pipeline()
        nodes = chunking.process(documents)
        
        print(f"✓ Created {len(nodes)} chunks with embeddings")
        
        # Step 3: Index in vector DB
        print("\n" + "="*60)
        print("PHASE 3: Vector Database Indexing")
        print("="*60)
        
        indexer = self._get_indexer()
        
        if recreate:
            indexer.create_collection(recreate=True)
        
        indexed_count = indexer.index_nodes(nodes)
        
        print(f"✓ Indexed {indexed_count} nodes in vector database")
        
        # Get collection info
        info = indexer.get_collection_info()
        
        # Build LlamaIndex index
        from src.indexing import VectorIndexBuilder
        builder = VectorIndexBuilder(indexer, self._get_embed_model())
        self._index = builder.build_index(nodes)
        
        print(f"✓ Built LlamaIndex VectorStoreIndex")
        
        stats = {
            "documents_ingested": len(documents),
            "chunks_created": len(nodes),
            "nodes_indexed": indexed_count,
            "collection_info": info
        }
        
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        print(f"Documents: {stats['documents_ingested']}")
        print(f"Chunks: {stats['chunks_created']}")
        print(f"Indexed: {stats['nodes_indexed']}")
        
        return stats
    
    def query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            filters: Optional metadata filters
            evaluate: Whether to evaluate the response
            
        Returns:
            Query results including answer, sources, and evaluation
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # Step 1: Retrieve relevant context
        retriever = self._get_retriever()
        formatted_context, contexts, nodes = retriever.retrieve(question, filters)
        
        if not contexts:
            return {
                "answer": "I couldn't find relevant information in the documents to answer this question.",
                "sources": [],
                "contexts": [],
                "evaluation": None
            }
        
        # Step 2: Generate response
        generator = self._get_generator()
        result = generator.generate(question, formatted_context, contexts)
        
        # Step 3: Evaluate if enabled
        evaluation = None
        if evaluate and self.config.enable_evaluation:
            evaluator = self._get_evaluator()
            eval_result = evaluator.evaluate_single(
                question,
                result.answer,
                formatted_context
            )
            evaluation = {
                "faithfulness": eval_result.faithfulness_score,
                "relevancy": eval_result.relevancy_score,
                "hallucination_detected": eval_result.hallucination_detected,
                "passed": eval_result.passed
            }
        
        return {
            "answer": result.answer,
            "sources": result.sources,
            "contexts": [
                {
                    "text": ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text,
                    "source": ctx.source,
                    "type": ctx.content_type,
                    "score": ctx.score
                }
                for ctx in contexts
            ],
            "evaluation": evaluation
        }
    
    def interactive(self):
        """
        Run interactive query session.
        """
        # Suppress verbose logging for clean interactive output
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("src.indexing").setLevel(logging.WARNING)
        logging.getLogger("src.retrieval").setLevel(logging.WARNING)
        logging.getLogger("src.generation").setLevel(logging.WARNING)
        logging.getLogger("src.evaluation").setLevel(logging.WARNING)
        logging.getLogger("llama_index").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("__main__").setLevel(logging.WARNING)
        
        print("\n" + "="*60)
        print("LCA RAG Interactive Mode")
        print("="*60)
        print("Type your questions about LCA documents.")
        print("Commands: /quit, /sources, /eval, /help")
        print("="*60 + "\n")
        
        show_sources = True
        show_eval = True
        
        while True:
            try:
                question = input("\n📝 Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.startswith("/"):
                    cmd = question.lower()
                    if cmd in ["/quit", "/exit", "/q"]:
                        print("Goodbye!")
                        break
                    elif cmd == "/sources":
                        show_sources = not show_sources
                        print(f"Sources display: {'ON' if show_sources else 'OFF'}")
                        continue
                    elif cmd == "/eval":
                        show_eval = not show_eval
                        print(f"Evaluation display: {'ON' if show_eval else 'OFF'}")
                        continue
                    elif cmd == "/help":
                        print("Commands:")
                        print("  /quit    - Exit interactive mode")
                        print("  /sources - Toggle source display")
                        print("  /eval    - Toggle evaluation display")
                        print("  /help    - Show this help")
                        continue
                    else:
                        print(f"Unknown command: {question}")
                        continue
                
                # Process query (silently)
                result = self.query(question, evaluate=show_eval)
                
                # Display answer
                print("\n📖 Answer:")
                print("-"*40)
                print(result["answer"])
                
                # Display sources
                if show_sources and result["sources"]:
                    print("\n" + "-"*40)
                    print("📚 Sources:")
                    print("-"*40)
                    for source in result["sources"]:
                        source_name = source.split("/")[-1].split("\\")[-1]
                        print(f"  • {source_name}")
                
                # Display evaluation
                if show_eval and result["evaluation"]:
                    eval_data = result["evaluation"]
                    print("\n" + "-"*40)
                    print("📊 Evaluation:")
                    print("-"*40)
                    print(f"  Faithfulness: {eval_data['faithfulness']:.2f}")
                    print(f"  Relevancy: {eval_data['relevancy']:.2f}")
                    status = "✅ PASSED" if eval_data["passed"] else "⚠️ NEEDS REVIEW"
                    print(f"  Status: {status}")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {e}")
    
    def test(self, sample_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run test queries to verify the pipeline.
        
        Args:
            sample_queries: Optional list of test queries
            
        Returns:
            Test results
        """
        if sample_queries is None:
            sample_queries = [
                "What is Life Cycle Assessment?",
                "What is the environmental impact of steel production?",
                "Explain the ReCiPe methodology.",
                "What are the phases of LCA?",
                "How do you calculate Global Warming Potential?"
            ]
        
        print("\n" + "="*60)
        print("PIPELINE TEST")
        print("="*60)
        
        results = []
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n--- Test {i}/{len(sample_queries)} ---")
            print(f"Query: {query}")
            
            try:
                result = self.query(query)
                
                print(f"Answer preview: {result['answer'][:150]}...")
                print(f"Sources: {len(result['sources'])}")
                
                if result["evaluation"]:
                    eval_data = result["evaluation"]
                    status = "✅" if eval_data["passed"] else "⚠️"
                    print(f"Evaluation: {status} (faith: {eval_data['faithfulness']:.2f})")
                
                results.append({
                    "query": query,
                    "success": True,
                    "evaluation": result["evaluation"]
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        # Summary
        success_count = sum(1 for r in results if r["success"])
        pass_count = sum(1 for r in results 
                        if r["success"] and r.get("evaluation", {}).get("passed", False))
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total queries: {len(sample_queries)}")
        print(f"Successful: {success_count}")
        print(f"Passed evaluation: {pass_count}")
        
        return {
            "total": len(sample_queries),
            "successful": success_count,
            "passed": pass_count,
            "results": results
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LCA RAG Pipeline - Query your LCA documents with AI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="./LCA",
        help="Directory containing documents"
    )
    ingest_parser.add_argument(
        "--recreate", "-r",
        action="store_true",
        help="Recreate the index from scratch"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument(
        "question",
        type=str,
        help="Question to ask"
    )
    query_parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation"
    )
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive query mode")
    
    # Test command
    subparsers.add_parser("test", help="Run test queries")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create pipeline
    config = PipelineConfig()
    pipeline = LCARagPipeline(config)
    
    # Execute command
    if args.command == "ingest":
        pipeline.ingest(
            data_dir=Path(args.data_dir),
            recreate=args.recreate
        )
        
    elif args.command == "query":
        result = pipeline.query(
            args.question,
            evaluate=not args.no_eval
        )
        
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result["answer"])
        
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"  • {source}")
        
        if result["evaluation"]:
            eval_data = result["evaluation"]
            status = "✅ PASSED" if eval_data["passed"] else "⚠️ NEEDS REVIEW"
            print(f"\nEvaluation: {status} (faithfulness: {eval_data['faithfulness']:.2f})")
            
    elif args.command == "interactive":
        pipeline.interactive()
        
    elif args.command == "test":
        pipeline.test()


if __name__ == "__main__":
    main()
