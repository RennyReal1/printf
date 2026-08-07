"""Layout-aware PDF corpus retrieval with cross-paper attribution."""

from .blocks import BlockType, Chunk, RawBlock, Unit
from .chunker import chunk_document
from .classify import classify_document
from .extract import extract_blocks
from .index import CorpusIndex
from .retrieve import build_context, resolve_citations, retrieve

__all__ = [
    "BlockType",
    "Chunk",
    "RawBlock",
    "Unit",
    "chunk_document",
    "classify_document",
    "extract_blocks",
    "CorpusIndex",
    "retrieve",
    "build_context",
    "resolve_citations",
]
