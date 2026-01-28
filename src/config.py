"""
Configuration settings for the LCA RAG Application.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    """Path configuration for the application."""
    
    # Base paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "LCA")
    
    # Storage paths
    vector_db_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "qdrant")
    parsed_docs_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "parsed")
    cache_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cache")
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.parsed_docs_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    
    # Text embedding model
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_dim: int = 384
    
    # Alternative larger model for better quality
    # text_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    # text_embedding_dim: int = 768
    
    # Device configuration
    device: str = "cuda"  # or "cpu" if no GPU
    
    # Batch processing
    batch_size: int = 32


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    # Chunk sizes
    chunk_size: int = 256  # Reduced from 512 for large documents
    chunk_overlap: int = 128
    # Separator for splitting
    separator: str = "\n\n"
    # Preserve structure boundaries
    preserve_tables: bool = True
    preserve_lists: bool = True


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    
    # Qdrant settings
    collection_name: str = "lca_documents"
    host: str = "localhost"
    port: int = 6333
    
    # Use local in-memory or on-disk mode for testing
    use_local: bool = True
    prefer_grpc: bool = False
    
    # Search settings
    top_k: int = 10
    similarity_threshold: float = 0.7


@dataclass
class LLMConfig:
    """LLM configuration."""
    
    # OpenRouter settings
    model_name: str = "google/gemini-2.0-flash-001"
    
    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 2048
    context_window: int = 4096
    
    # Request timeout
    request_timeout: float = 120.0


@dataclass
class ParsingConfig:
    """Document parsing configuration."""
    
    # Supported file extensions
    docling_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", 
        ".pptx", ".ppt", ".png", ".jpg", ".jpeg", 
        ".gif", ".bmp", ".tiff"
    ])
    
    mindmap_extensions: List[str] = field(default_factory=lambda: [".xmind"])
    
    # Docling settings
    enable_ocr: bool = True
    enable_table_extraction: bool = True
    enable_image_extraction: bool = True
    
    # Output format
    export_format: str = "markdown"  # or "json"


@dataclass
class RAGConfig:
    """Main RAG configuration combining all configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    
    # Evaluation settings
    enable_evaluation: bool = True
    faithfulness_threshold: float = 0.9


# Default configuration instance
def get_config() -> RAGConfig:
    """Get the default configuration."""
    return RAGConfig()
