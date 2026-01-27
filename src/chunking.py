"""
Chunking and Embedding Module for LCA RAG Application.

This module handles:
- Smart chunking of documents while preserving structure
- Text embeddings using Sentence Transformers
- Handling different content types appropriately
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# LlamaIndex imports
from llama_index.core.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 256  # Reduced from 512 for large documents
    chunk_overlap: int = 128
    separator: str = "\n\n"
    preserve_structure: bool = True


class StructureAwareChunker:
    """
    Chunker that preserves document structure.
    
    Handles:
    - Tables: kept intact or split at row boundaries
    - Mind maps: split at branch boundaries
    - Regular text: sentence-based splitting
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Initialize sentence splitter for regular text
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator=self.config.separator
        )
        
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Chunk a list of documents into nodes.
        
        Args:
            documents: List of LlamaIndex Documents
            
        Returns:
            List of TextNode objects
        """
        all_nodes = []
        
        for doc in tqdm(documents, desc="Chunking documents"):
            nodes = self.chunk_document(doc)
            all_nodes.extend(nodes)
            
        logger.info(f"Created {len(all_nodes)} chunks from {len(documents)} documents")
        return all_nodes
    
    def chunk_document(self, document: Document) -> List[TextNode]:
        """
        Chunk a single document based on its content type.
        
        Args:
            document: LlamaIndex Document
            
        Returns:
            List of TextNode objects
        """
        content_type = document.metadata.get("content_type", "text")
        
        if content_type == "table":
            return self._chunk_table(document)
        elif content_type == "mindmap":
            return self._chunk_mindmap(document)
        elif content_type == "calculation":
            return self._chunk_calculation(document)
        elif content_type == "image":
            return self._chunk_image(document)
        else:
            return self._chunk_text(document)
    
    def _chunk_text(self, document: Document) -> List[TextNode]:
        """Chunk regular text using sentence splitting."""
        nodes = self.sentence_splitter.get_nodes_from_documents([document])
        
        # Ensure metadata is preserved and add chunk info
        for i, node in enumerate(nodes):
            node.metadata.update({
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(nodes)
            })
            
        return nodes
    
    def _chunk_table(self, document: Document) -> List[TextNode]:
        """
        Chunk table content while preserving structure.
        
        Tables are either kept whole or split at row boundaries.
        """
        text = document.text
        lines = text.split("\n")
        
        # If table is small enough, keep it whole
        if len(text) <= self.config.chunk_size * 1.5:
            node = TextNode(
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunking_strategy": "table_whole"
                }
            )
            return [node]
        
        # Split at row boundaries
        nodes = []
        current_chunk = []
        current_length = 0
        header = None
        
        for i, line in enumerate(lines):
            # Detect and preserve header
            if i == 0 and "|" in line:
                header = line
                
            line_length = len(line)
            
            if current_length + line_length > self.config.chunk_size:
                # Create node from current chunk
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    # Prepend header to each chunk
                    if header and header not in chunk_text:
                        chunk_text = header + "\n" + chunk_text
                        
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_index": len(nodes),
                            "chunking_strategy": "table_rows"
                        }
                    )
                    nodes.append(node)
                    
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Handle remaining content
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            if header and header not in chunk_text:
                chunk_text = header + "\n" + chunk_text
                
            node = TextNode(
                text=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": len(nodes),
                    "chunking_strategy": "table_rows"
                }
            )
            nodes.append(node)
        
        # Update total chunks
        for node in nodes:
            node.metadata["total_chunks"] = len(nodes)
            
        return nodes
    
    def _chunk_mindmap(self, document: Document) -> List[TextNode]:
        """
        Chunk mind map content at branch boundaries.
        
        Preserves hierarchical relationships.
        """
        text = document.text
        
        # If it's JSON format, chunk differently
        if document.metadata.get("format") == "json":
            # For JSON, keep as single chunk for structure
            node = TextNode(
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunking_strategy": "mindmap_json"
                }
            )
            return [node]
        
        # For flattened text, split at top-level branches
        lines = text.split("\n")
        nodes = []
        current_branch = []
        
        for line in lines:
            # Detect top-level items (no leading whitespace)
            if line.strip().startswith("•") and not line.startswith(" "):
                if current_branch:
                    branch_text = "\n".join(current_branch)
                    if len(branch_text) > 50:  # Skip very small chunks
                        node = TextNode(
                            text=branch_text,
                            metadata={
                                **document.metadata,
                                "chunk_index": len(nodes),
                                "chunking_strategy": "mindmap_branch"
                            }
                        )
                        nodes.append(node)
                current_branch = [line]
            else:
                current_branch.append(line)
        
        # Handle last branch
        if current_branch:
            branch_text = "\n".join(current_branch)
            if len(branch_text) > 50:
                node = TextNode(
                    text=branch_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": len(nodes),
                        "chunking_strategy": "mindmap_branch"
                    }
                )
                nodes.append(node)
        
        # If no branches detected, use regular chunking
        if not nodes:
            return self._chunk_text(document)
        
        # Update total chunks
        for node in nodes:
            node.metadata["total_chunks"] = len(nodes)
            
        return nodes
    
    def _chunk_calculation(self, document: Document) -> List[TextNode]:
        """
        Chunk calculation content.
        
        Keeps formulas and their context together.
        """
        # For calculations, try to keep formulas intact
        text = document.text
        
        # Keep small calculations whole
        if len(text) <= self.config.chunk_size * 1.5:
            node = TextNode(
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunking_strategy": "calculation_whole"
                }
            )
            return [node]
        
        # Otherwise use sentence splitting
        return self._chunk_text(document)
    
    def _chunk_image(self, document: Document) -> List[TextNode]:
        """
        Handle image content.
        
        Images with captions are kept as single nodes.
        """
        node = TextNode(
            text=document.text,
            metadata={
                **document.metadata,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "image_description"
            }
        )
        return [node]


class EmbeddingManager:
    """
    Manages embedding generation for documents.
    
    Uses Sentence Transformers for text embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run on ("cuda" or "cpu")
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._embed_model = None
        
    def _get_model(self) -> HuggingFaceEmbedding:
        """Lazy load the embedding model."""
        if self._embed_model is None:
            try:
                self._embed_model = HuggingFaceEmbedding(
                    model_name=self.model_name,
                    device=self.device
                )
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load on {self.device}, falling back to CPU: {e}")
                self._embed_model = HuggingFaceEmbedding(
                    model_name=self.model_name,
                    device="cpu"
                )
        return self._embed_model
    
    def get_embedding_model(self) -> HuggingFaceEmbedding:
        """Get the embedding model instance."""
        return self._get_model()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        model = self._get_model()
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = [model.get_text_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        Add embeddings to nodes.
        
        Args:
            nodes: List of TextNode objects
            
        Returns:
            Nodes with embeddings added
        """
        model = self._get_model()
        
        for i, node in enumerate(tqdm(nodes, desc="Embedding nodes")):
            if node.embedding is None:
                node.embedding = model.get_text_embedding(node.text)
                
        logger.info(f"Generated embeddings for {len(nodes)} nodes")
        return nodes


class ChunkingEmbeddingPipeline:
    """
    Combined pipeline for chunking and embedding.
    """
    
    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        """
        Initialize the pipeline.
        
        Args:
            chunking_config: Configuration for chunking
            model_name: Embedding model name
            device: Device for embeddings
        """
        self.chunker = StructureAwareChunker(chunking_config)
        self.embedding_manager = EmbeddingManager(
            model_name=model_name,
            device=device
        )
        
    def process(self, documents: List[Document]) -> List[TextNode]:
        """
        Chunk documents and generate embeddings.
        
        Args:
            documents: List of LlamaIndex Documents
            
        Returns:
            List of TextNode objects with embeddings
        """
        # Chunk documents
        logger.info("Starting chunking...")
        nodes = self.chunker.chunk_documents(documents)
        
        # Generate embeddings
        logger.info("Starting embedding generation...")
        nodes = self.embedding_manager.embed_nodes(nodes)
        
        return nodes


def main():
    """Test the chunking and embedding pipeline."""
    # Create sample documents for testing
    sample_docs = [
        Document(
            text="""Life Cycle Assessment (LCA) is a methodology for assessing environmental impacts 
            associated with all the stages of the life cycle of a commercial product, process, or service.
            
            Key stages include:
            1. Raw material extraction
            2. Materials processing
            3. Manufacturing
            4. Distribution
            5. Use
            6. End-of-life disposal or recycling
            
            LCA helps identify opportunities to improve environmental performance throughout the product lifecycle.""",
            metadata={
                "source": "sample_lca.pdf",
                "content_type": "text",
                "file_type": ".pdf"
            }
        ),
        Document(
            text="""| Material | GWP (kg CO2 eq) | Energy (MJ) |
|----------|----------------|-------------|
| Steel    | 2.1            | 25          |
| Aluminum | 8.1            | 155         |
| Copper   | 3.0            | 42          |
| Plastic  | 3.5            | 78          |""",
            metadata={
                "source": "materials_table.xlsx",
                "content_type": "table",
                "file_type": ".xlsx"
            }
        ),
        Document(
            text="""• LCA Framework
  • Goal and Scope Definition
    • System Boundaries
    • Functional Unit
  • Life Cycle Inventory (LCI)
    • Data Collection
    • Mass Balance
  • Life Cycle Impact Assessment (LCIA)
    • Impact Categories
    • Characterization
  • Interpretation
    • Sensitivity Analysis
    • Conclusions""",
            metadata={
                "source": "lca_framework.xmind",
                "content_type": "mindmap",
                "file_type": ".xmind",
                "format": "flattened_text"
            }
        )
    ]
    
    # Initialize pipeline
    config = ChunkingConfig(
        chunk_size=256,  # Smaller for testing
        chunk_overlap=64
    )
    pipeline = ChunkingEmbeddingPipeline(
        chunking_config=config,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"  # Use CPU for testing
    )
    
    # Process documents
    print("\n=== Processing Documents ===")
    nodes = pipeline.process(sample_docs)
    
    # Show results
    print(f"\n=== Results ===")
    print(f"Total nodes created: {len(nodes)}")
    
    for i, node in enumerate(nodes):
        print(f"\n--- Node {i+1} ---")
        print(f"Content type: {node.metadata.get('content_type')}")
        print(f"Chunking strategy: {node.metadata.get('chunking_strategy')}")
        print(f"Text preview: {node.text[:150]}...")
        print(f"Embedding dimensions: {len(node.embedding) if node.embedding else 'None'}")


if __name__ == "__main__":
    main()
