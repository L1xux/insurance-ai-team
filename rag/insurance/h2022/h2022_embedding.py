"""
H2022 임베딩 래퍼
=========================
Author: Jin
Date: 2025.10.12
Version: 1.1

Description:
H2022 보험 문서용 임베딩 래퍼입니다.
"""
from typing import List

from base.rag.embedding_base import EmbeddingBase
from config.logging_config import logger


class H2022Embedding(EmbeddingBase):
    """H2022 전용 임베딩 클래스 - DI 패턴 적용"""
    
    def __init__(
        self,
        embedder: EmbeddingBase,
    ):
        """
        H2022 임베딩 초기화
        
        Args:
            embedder: 사용할 임베딩 모델 (EmbeddingBase 구현체)
                     예: OpenAIEmbedder, OllamaEmbedder 등
        
        """
        super().__init__(
            model_name=f"H2022-{embedder.model_name}",
            dimension=embedder.dimension
        )
        
        self.embedder = embedder
        
        logger.info(
            f"[H2022Embedding] H2022 임베딩 초기화 완료 "
            f"(embedder={embedder.model_name}, dimension={embedder.dimension})"
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트를 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        return self.embedder.embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 벡터로 변환 (배치 처리)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        return self.embedder.embed_texts(texts)
