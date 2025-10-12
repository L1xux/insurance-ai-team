"""
RAG 베이스 클래스 모듈
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
RAG(Retrieval-Augmented Generation) 시스템의 핵심 구성 요소들에 대한 추상 베이스 클래스를 제공합니다.
각 클래스는 ABC를 상속받아 일관된 인터페이스를 정의하며, 구현체는 이를 상속받아 확장합니다.
"""

from base.rag.embedding_base import EmbeddingBase
from base.rag.document_loader_base import DocumentLoaderBase, Document
from base.rag.preprocessor_base import PreprocessorBase, Chunk
from base.rag.vector_store_base import VectorStoreBase
from base.rag.indexer_base import IndexerBase
from base.rag.retriever_base import RetrieverBase, SearchResult
from base.rag.reranker_base import RerankerBase, RerankResult


__all__ = [
    # 베이스 클래스
    'EmbeddingBase',
    'DocumentLoaderBase',
    'PreprocessorBase',
    'VectorStoreBase',
    'IndexerBase',
    'RetrieverBase',
    'RerankerBase',
    
    # 데이터 클래스
    'Document',
    'Chunk',
    'SearchResult',
    'RerankResult',
]

