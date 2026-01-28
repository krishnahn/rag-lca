"""
Document ingestion module for the LCA RAG Application.
Handles parsing of various document formats using Docling and xmindparser.
"""

import logging
import warnings
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Suppress noisy OCR warnings
warnings.filterwarnings("ignore", message=".*RapidOCR.*")
warnings.filterwarnings("ignore", message=".*text detection result is empty.*")
logging.getLogger("docling.models.stages.ocr").setLevel(logging.ERROR)
logging.getLogger("rapidocr").setLevel(logging.ERROR)

from llama_index.core.schema import Document

from .config import RAGConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Represents a parsed document with content and metadata."""
    
    content: str
    metadata: Dict[str, Any]
    source_path: Path
    doc_type: str
    
    def to_llama_document(self) -> Document:
        """Convert to LlamaIndex Document format."""
        return Document(
            text=self.content,
            metadata={
                **self.metadata,
                "source": str(self.source_path),
                "doc_type": self.doc_type
            }
        )


class DoclingParser:
    """Parser for documents using Docling library."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._converter = None
        self._ocr_converter = None  # Lazy-initialized OCR fallback converter
        self._initialized = False
        self._ocr_initialized = False
        self._init_lock = threading.Lock()  # Thread-safe initialization
        self._ocr_lock = threading.Lock()  # Thread-safe OCR initialization
    
    def _init_converter(self):
        """Lazy initialization of Docling converter to avoid import overhead."""
        if self._initialized:
            return
        
        # Thread-safe initialization
        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return
            
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat
                
                # Configure PDF pipeline - DISABLE OCR to avoid empty result warnings
                # Most PDFs have embedded text, so OCR is not needed
                pdf_pipeline_options = PdfPipelineOptions()
                pdf_pipeline_options.do_ocr = False  # Disable OCR for text-based PDFs
                pdf_pipeline_options.do_table_structure = self.config.parsing.enable_table_extraction
                
                # Create converter with optimized settings
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pdf_pipeline_options
                        )
                    }
                )
                self._initialized = True
                logger.info("Docling converter initialized successfully (OCR disabled for text PDFs)")
                
            except ImportError as e:
                logger.error(f"Failed to import Docling: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Docling converter: {e}")
                raise
    
    def parse(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a document using Docling."""
        self._init_converter()
        
        try:
            # Convert the document
            result = self._converter.convert(str(file_path))
            
            # Export to markdown format
            content = result.document.export_to_markdown()
            
            # Handle empty content
            if not content or not content.strip():
                logger.warning(f"No text content extracted from {file_path.name}")
                # Try with OCR as fallback for scanned documents
                content = self._try_ocr_fallback(file_path)
                if not content:
                    return None
            
            # Extract metadata using correct Docling v2 API
            metadata = {
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
                "page_count": len(result.pages) if hasattr(result, 'pages') else 1,
                "title": getattr(result.document, 'name', file_path.stem) or file_path.stem,
            }
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                source_path=file_path,
                doc_type="docling"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            return None
    
    def _try_ocr_fallback(self, file_path: Path) -> Optional[str]:
        """Try OCR extraction as fallback for scanned documents."""
        try:
            # Lazy initialize OCR converter (reused across calls)
            if not self._ocr_initialized:
                # Thread-safe initialization
                with self._ocr_lock:
                    # Double-check after acquiring lock
                    if self._ocr_initialized:
                        # Already initialized by another thread
                        pass
                    else:
                        from docling.document_converter import DocumentConverter, PdfFormatOption
                        from docling.datamodel.pipeline_options import PdfPipelineOptions
                        from docling.datamodel.base_models import InputFormat
                    
                        pdf_pipeline_options = PdfPipelineOptions()
                        pdf_pipeline_options.do_ocr = True
                        pdf_pipeline_options.do_table_structure = False  # Simplify for OCR
                        
                        self._ocr_converter = DocumentConverter(
                            format_options={
                                InputFormat.PDF: PdfFormatOption(
                                    pipeline_options=pdf_pipeline_options
                                )
                            }
                        )
                        self._ocr_initialized = True
            
            result = self._ocr_converter.convert(str(file_path))
            content = result.document.export_to_markdown()
            
            if content and content.strip():
                logger.info(f"OCR fallback successful for {file_path.name}")
                return content
            
            return None
            
        except Exception as e:
            logger.debug(f"OCR fallback failed for {file_path.name}: {e}")
            return None


class XMindParser:
    """Parser for XMind mindmap files."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def parse(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse an XMind file."""
        try:
            import xmindparser
            
            # Parse the XMind file
            xmind_data = xmindparser.xmind_to_dict(str(file_path))
            
            # Convert to markdown-like text
            content = self._xmind_to_text(xmind_data)
            
            if not content or not content.strip():
                logger.warning(f"No content extracted from XMind file: {file_path.name}")
                return None
            
            metadata = {
                "filename": file_path.name,
                "file_type": ".xmind",
                "file_size": file_path.stat().st_size,
            }
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                source_path=file_path,
                doc_type="xmind"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse XMind file {file_path.name}: {e}")
            return None
    
    def _xmind_to_text(self, data: List[Dict], level: int = 0) -> str:
        """Recursively convert XMind data to text."""
        lines = []
        indent = "  " * level
        
        for item in data:
            if isinstance(item, dict):
                # Handle sheet/topic structure
                if "topic" in item:
                    topic = item["topic"]
                    title = topic.get("title", "")
                    lines.append(f"{indent}{'#' * (level + 1)} {title}")
                    
                    # Process subtopics
                    if "topics" in topic:
                        lines.append(self._xmind_to_text(topic["topics"], level + 1))
                
                elif "title" in item:
                    title = item.get("title", "")
                    lines.append(f"{indent}{'#' * (level + 1)} {title}")
                    
                    if "topics" in item:
                        lines.append(self._xmind_to_text(item["topics"], level + 1))
        
        return "\n".join(lines)


class TextFileParser:
    """Simple parser for plain text files."""
    
    # Maximum file size to read into memory (100 MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def parse(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a plain text file."""
        try:
            # Check file size to prevent memory issues
            file_size = file_path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                logger.warning(
                    f"Skipping {file_path.name}: file size {file_size / 1024 / 1024:.1f}MB "
                    f"exceeds maximum {self.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
                )
                return None
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.error(f"Failed to decode {file_path.name} with any encoding")
                return None
            
            metadata = {
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
            }
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                source_path=file_path,
                doc_type="text"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse text file {file_path.name}: {e}")
            return None


class DocumentIngester:
    """Main class for ingesting documents from various sources."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_config()
        self.docling_parser = DoclingParser(self.config)
        self.xmind_parser = XMindParser(self.config)
        self.text_parser = TextFileParser(self.config)
        
        # Supported extensions mapping
        self.extension_map = {
            **{ext: "docling" for ext in self.config.parsing.docling_extensions},
            **{ext: "xmind" for ext in self.config.parsing.mindmap_extensions},
            ".txt": "text",
            ".md": "text",
            ".csv": "text",
        }
    
    def discover_files(self, directory: Path) -> List[Path]:
        """Discover all supported files in a directory recursively."""
        files = []
        supported_extensions = set(self.extension_map.keys())
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        logger.info(f"Discovered {len(files)} files in {directory}")
        return files
    
    def _parse_single_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a single file based on its type."""
        ext = file_path.suffix.lower()
        parser_type = self.extension_map.get(ext)
        
        if parser_type == "docling":
            return self.docling_parser.parse(file_path)
        elif parser_type == "xmind":
            return self.xmind_parser.parse(file_path)
        elif parser_type == "text":
            return self.text_parser.parse(file_path)
        else:
            logger.warning(f"No parser for extension: {ext}")
            return None
    
    def ingest_directory(
        self, 
        directory: Path,
        max_workers: int = 4,
        show_progress: bool = True
    ) -> List[Document]:
        """
        Ingest all documents from a directory (parallelized).
        
        Args:
            directory: Path to the directory containing documents
            max_workers: Maximum number of parallel workers
            show_progress: Whether to show a progress bar
            
        Returns:
            List of LlamaIndex Document objects
        """
        files = self.discover_files(directory)
        
        if not files:
            logger.warning(f"No supported files found in {directory}")
            return []
        
        documents = []
        failed_files = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(self._parse_single_file, f): f 
                for f in files
            }
            
            # Create progress bar
            iterator = as_completed(future_to_file)
            if show_progress:
                iterator = tqdm(
                    iterator, 
                    total=len(files), 
                    desc="Ingesting documents"
                )
            
            for future in iterator:
                file_path = future_to_file[future]
                try:
                    parsed_doc = future.result()
                    if parsed_doc:
                        documents.append(parsed_doc.to_llama_document())
                    else:
                        failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    failed_files.append(file_path)
        
        # Log summary
        logger.info(f"Successfully ingested {len(documents)} documents")
        if failed_files:
            logger.warning(f"Failed to parse {len(failed_files)} files:")
            for f in failed_files[:10]:  # Show first 10 failures
                logger.warning(f"  - {f.name}")
            if len(failed_files) > 10:
                logger.warning(f"  ... and {len(failed_files) - 10} more")
        
        return documents
    
    def ingest_file(self, file_path: Path) -> Optional[Document]:
        """Ingest a single file."""
        parsed_doc = self._parse_single_file(file_path)
        if parsed_doc:
            return parsed_doc.to_llama_document()
        return None


def ingest_documents(
    data_dir: Optional[Path] = None,
    config: Optional[RAGConfig] = None,
    max_workers: int = 4
) -> List[Document]:
    """
    Convenience function to ingest documents from a directory.
    
    Args:
        data_dir: Path to the data directory (defaults to config path)
        config: RAG configuration (defaults to standard config)
        max_workers: Number of parallel workers
        
    Returns:
        List of LlamaIndex Document objects
    """
    config = config or get_config()
    data_dir = data_dir or config.paths.data_dir
    
    ingester = DocumentIngester(config)
    return ingester.ingest_directory(Path(data_dir), max_workers=max_workers)


# Alias for backward compatibility with pipeline.py
DataIngestionPipeline = DocumentIngester
