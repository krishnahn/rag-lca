# LCA RAG Application

A production-grade Retrieval-Augmented Generation (RAG) system for Life Cycle Assessment (LCA) documents.

## Features

- **Multi-format Document Ingestion**: Parse PDFs, Word, Excel, PowerPoint, images, and mind maps
- **Intelligent Chunking**: Structure-aware chunking that preserves tables, calculations, and hierarchies
- **Vector Database**: Qdrant-based storage with metadata filtering
- **LLM Integration**: Ollama with Llama 3.2 for grounded prompts
- **Evaluation**: Faithfulness scoring and hallucination detection

### Changes Implemented

| Feature | Description |
|---------|-------------|
| LangSmith Integration | Added `LangSmithEvaluator` class in `src/evaluation.py` for tracing and evaluation. Set `LANGSMITH_API_KEY` env var to enable. |
| Semantic Chunking | Added `SemanticChunker` class in `src/chunking.py` that splits documents based on embedding similarity for more coherent chunks. |
| FlashRank Reranker | Added `FlashRankReranker` in `src/retrieval.py` using `ms-marco-MiniLM-L-12-v2` for ultra-fast (<50ms) reranking. |
| Faster Inference | Optimized `src/generation.py`: reduced `num_predict=256`, `num_ctx=2048`, `temperature=0.1`, shorter prompts, aggressive timeout (30s). |

## Project Structure

```
RAG - Arantree project/
├── LCA/                    # Your LCA documents
├── data/                   # Generated data (vector DB, cache)
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── ingestion.py       # Document parsing (Docling + xmindparser)
│   ├── chunking.py        # Chunking and embedding
│   ├── indexing.py        # Vector database operations
│   ├── retrieval.py       # Query processing and retrieval
│   ├── generation.py      # LLM response generation
│   ├── evaluation.py      # Answer quality evaluation
│   └── pipeline.py        # Main orchestration
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Ollama

Install and run Ollama:

```bash
# Install Ollama from https://ollama.ai

# Pull the Llama 3.2 model
ollama pull llama3.2

# Start Ollama server (if not already running)
ollama serve
```

## Usage

### Quick Start

```bash
# Step 1: Ingest your LCA documents
python -m src.pipeline ingest --data-dir ./LCA

# Step 2: Query the system
python -m src.pipeline query "What is the environmental impact of steel production?"

# Step 3: Interactive mode
python -m src.pipeline interactive
```

### Commands

| Command | Description |
|---------|-------------|
| `ingest` | Parse and index documents from a directory |
| `query` | Ask a single question |
| `interactive` | Start interactive Q&A session |
| `test` | Run test queries to verify the pipeline |

### Examples

```bash
# Ingest documents (recreate index from scratch)
python -m src.pipeline ingest --data-dir ./LCA --recreate

# Query with evaluation disabled
python -m src.pipeline query "Explain the ReCiPe methodology" --no-eval

# Run test queries
python -m src.pipeline test
```

### Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit interactive mode |
| `/sources` | Toggle source display |
| `/eval` | Toggle evaluation display |
| `/help` | Show help |

## Architecture

### Phase 1: Document Ingestion
- Scans directories for supported files
- Uses **Docling** for PDFs, Word, Excel, PowerPoint, images
- Uses **xmindparser** for mind map files
- Outputs LlamaIndex Documents with metadata

### Phase 2: Chunking
- Structure-aware chunking preserving:
  - Table boundaries
  - Mind map hierarchies
  - Calculation contexts
- Configurable chunk size (default: 256 tokens, 64 overlap)

### Phase 3: Embedding
- Uses Sentence Transformers (all-MiniLM-L6-v2)
- 384-dimensional embeddings
- GPU acceleration supported

### Phase 4: Indexing
- Qdrant vector database (local file-based or server)
- Metadata filtering support
- Hybrid search capabilities

### Phase 5: Retrieval
- Query embedding and expansion
- Top-k retrieval with reranking
- Multi-source context aggregation
- Metadata-based filtering

### Phase 6: Generation
- Ollama LLM integration (Llama 3.2)
- Custom prompts for grounded responses
- Source citation in answers
- Multi-document synthesis

### Phase 7: Evaluation
- Faithfulness scoring (heuristic + LLM-based)
- Hallucination detection
- RAGAS integration for comprehensive evaluation

## Configuration

Edit `src/config.py` or pass parameters via command line:

```python
@dataclass
class PipelineConfig:
    # Paths
    data_dir: Path = Path("./LCA")
    vector_db_path: Path = Path("./data/qdrant")
    
    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda"
    
    # Chunking
    chunk_size: int = 256
    chunk_overlap: int = 64
    
    # LLM (Ollama)
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.1
    
    # Retrieval
    top_k: int = 10
    similarity_threshold: float = 0.5
```

## Supported File Types

| Type | Extensions | Parser |
|------|------------|--------|
| PDF | `.pdf` | Docling / pypdf |
| Word | `.docx`, `.doc` | Docling / python-docx |
| Excel | `.xlsx`, `.xls` | Docling / pandas |
| PowerPoint | `.pptx`, `.ppt` | Docling / python-pptx |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff` | Docling / PIL |
| Mind Maps | `.xmind` | xmindparser |

## Programmatic Usage

```python
from src.pipeline import LCARagPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(
    data_dir=Path("./LCA"),
    llm_model="llama3.2"
)
pipeline = LCARagPipeline(config)

# Ingest documents
stats = pipeline.ingest()
print(f"Indexed {stats['nodes_indexed']} chunks")

# Query
result = pipeline.query("What is LCA?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

## Testing Individual Modules

```bash
# Test ingestion
python -m src.ingestion

# Test chunking
python -m src.chunking

# Test indexing
python -m src.indexing

# Test retrieval
python -m src.retrieval

# Test generation
python -m src.generation

# Test evaluation
python -m src.evaluation
```

## Troubleshooting

### Ollama not available
```
Error: Ollama is not available
```
Solution: 
1. Ensure Ollama is installed: https://ollama.ai
2. Start Ollama server: `ollama serve`
3. Pull the model: `ollama pull llama3.2`

### CUDA out of memory
Solution: Set `embedding_device: str = "cpu"` in config

### Import errors
Solution: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

Internal use - Arantree Consulting

## Author

Developed for LCA document analysis and querying.
