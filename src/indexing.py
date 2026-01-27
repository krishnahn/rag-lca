"""
Vector Database Indexing Module for LCA RAG Application.

This module handles:
- Creating and managing Qdrant vector database
- Storing document chunks with embeddings
- Metadata-based filtering support
- Hybrid search capabilities
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range
)

# LlamaIndex imports
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    collection_name: str = "lca_documents"
    host: str = "localhost"
    port: int = 6333
    use_local: bool = True  # Use local file-based storage
    local_path: str = "./data/qdrant"
    embedding_dim: int = 384
    distance_metric: str = "Cosine"


class QdrantIndexer:
    """
    Manages Qdrant vector database operations.
    
    Supports both local (file-based) and server modes.
    """
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """
        Initialize the Qdrant indexer.
        
        Args:
            config: Vector database configuration
        """
        self.config = config or VectorDBConfig()
        self._client = None
        self._vector_store = None
        
    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            if self.config.use_local:
                # Use local file-based storage
                local_path = Path(self.config.local_path)
                local_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(local_path))
                logger.info(f"Connected to local Qdrant at {local_path}")
            else:
                # Connect to Qdrant server
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port
                )
                logger.info(f"Connected to Qdrant server at {self.config.host}:{self.config.port}")
        return self._client
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the vector collection if it doesn't exist.
        
        Args:
            recreate: If True, delete and recreate existing collection
            
        Returns:
            True if collection was created/exists
        """
        client = self._get_client()
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.config.collection_name in collection_names:
            if recreate:
                logger.info(f"Recreating collection: {self.config.collection_name}")
                client.delete_collection(self.config.collection_name)
            else:
                logger.info(f"Collection already exists: {self.config.collection_name}")
                return True
        
        # Create collection
        distance = Distance.COSINE if self.config.distance_metric == "Cosine" else Distance.EUCLID
        
        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.embedding_dim,
                distance=distance
            )
        )
        
        logger.info(f"Created collection: {self.config.collection_name}")
        return True
    
    def index_nodes(self, nodes: List[TextNode], batch_size: int = 100) -> int:
        """
        Index nodes into the vector database.
        
        Args:
            nodes: List of TextNode objects with embeddings
            batch_size: Batch size for upserting
            
        Returns:
            Number of nodes indexed
        """
        client = self._get_client()
        
        # Ensure collection exists
        self.create_collection()
        
        # Prepare points
        points = []
        for node in tqdm(nodes, desc="Preparing nodes for indexing"):
            if node.embedding is None:
                logger.warning(f"Node {node.id_} has no embedding, skipping")
                continue
                
            # Generate unique ID if not present
            node_id = node.id_ if node.id_ else str(uuid.uuid4())
            
            # Prepare payload (metadata)
            payload = {
                "text": node.text,
                "node_id": node_id,
                **node.metadata
            }
            
            point = PointStruct(
                id=str(uuid.uuid4()),  # Qdrant point ID
                vector=node.embedding,
                payload=payload
            )
            points.append(point)
        
        # Batch upsert
        total_indexed = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )
            total_indexed += len(batch)
            
        logger.info(f"Indexed {total_indexed} nodes into collection {self.config.collection_name}")
        return total_indexed
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        client = self._get_client()
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            qdrant_filter = Filter(must=conditions)
        
        # Perform search
        results = client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })
            
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        client = self._get_client()
        
        try:
            info = client.get_collection(self.config.collection_name)
            return {
                "name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        client = self._get_client()
        
        try:
            client.delete_collection(self.config.collection_name)
            logger.info(f"Deleted collection: {self.config.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_vector_store(self) -> QdrantVectorStore:
        """Get LlamaIndex compatible vector store."""
        if self._vector_store is None:
            client = self._get_client()
            self.create_collection()
            
            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name
            )
            
        return self._vector_store


class VectorIndexBuilder:
    """
    Builds and manages the vector index using LlamaIndex.
    """
    
    def __init__(
        self,
        indexer: QdrantIndexer,
        embed_model: Any
    ):
        """
        Initialize the index builder.
        
        Args:
            indexer: QdrantIndexer instance
            embed_model: Embedding model for queries
        """
        self.indexer = indexer
        self.embed_model = embed_model
        self._index = None
        
    def build_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """
        Build a LlamaIndex VectorStoreIndex from nodes.
        
        Args:
            nodes: List of TextNode objects with embeddings
            
        Returns:
            VectorStoreIndex instance
        """
        # Get vector store
        vector_store = self.indexer.get_vector_store()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Build index
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        logger.info(f"Built vector index with {len(nodes)} nodes")
        return self._index
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load existing index from vector store.
        
        Returns:
            VectorStoreIndex if exists, None otherwise
        """
        try:
            vector_store = self.indexer.get_vector_store()
            
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
            
            logger.info("Loaded existing vector index")
            return self._index
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return None
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the current index."""
        return self._index


def main():
    """Test the vector database indexing."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    # Initialize components
    config = VectorDBConfig(
        collection_name="lca_test",
        use_local=True,
        local_path="./data/qdrant_test",
        embedding_dim=384
    )
    
    indexer = QdrantIndexer(config)
    
    # Create embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Create sample nodes with embeddings
    sample_texts = [
        "Life Cycle Assessment (LCA) evaluates environmental impacts across a product's lifecycle.",
        "Carbon footprint includes all greenhouse gas emissions from production to disposal.",
        "Steel production has a Global Warming Potential of approximately 2.1 kg CO2 equivalent.",
        "ReCiPe methodology provides characterization factors for impact assessment.",
        "Functional unit defines the quantified performance of a product system."
    ]
    
    print("\n=== Creating Sample Nodes ===")
    nodes = []
    for i, text in enumerate(sample_texts):
        embedding = embed_model.get_text_embedding(text)
        node = TextNode(
            text=text,
            metadata={
                "source": f"sample_{i}.pdf",
                "content_type": "text",
                "topic": "LCA" if i < 3 else "methodology"
            }
        )
        node.embedding = embedding
        nodes.append(node)
    
    print(f"Created {len(nodes)} sample nodes")
    
    # Index nodes
    print("\n=== Indexing Nodes ===")
    indexer.create_collection(recreate=True)
    indexed_count = indexer.index_nodes(nodes)
    print(f"Indexed {indexed_count} nodes")
    
    # Get collection info
    print("\n=== Collection Info ===")
    info = indexer.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test search
    print("\n=== Testing Search ===")
    query = "What is the environmental impact of steel?"
    query_embedding = embed_model.get_text_embedding(query)
    
    results = indexer.search(query_embedding, top_k=3)
    print(f"Query: {query}")
    print(f"Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Text: {result['text'][:80]}...")
    
    # Test filtered search
    print("\n=== Testing Filtered Search ===")
    results = indexer.search(
        query_embedding,
        top_k=3,
        filters={"content_type": "text", "topic": "LCA"}
    )
    print(f"Filtered results (topic=LCA):")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Text: {result['text'][:80]}...")
    
    # Build LlamaIndex index
    print("\n=== Building LlamaIndex Index ===")
    builder = VectorIndexBuilder(indexer, embed_model)
    index = builder.build_index(nodes)
    print(f"Index built successfully")
    
    # Cleanup
    print("\n=== Cleanup ===")
    indexer.delete_collection()
    print("Test collection deleted")


if __name__ == "__main__":
    main()
