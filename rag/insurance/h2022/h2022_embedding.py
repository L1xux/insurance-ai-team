"""
H2022 임베딩 래퍼
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
H2022 보험 문서용 임베딩 래퍼입니다.
OpenAI 임베딩을 사용하여 h2022 전용 설정을 제공합니다.
"""
import os
from typing import List, Optional

from dotenv import load_dotenv

from base.rag.embedding_base import EmbeddingBase
from rag.common.openai_embedder import OpenAIEmbedder
from config.logging_config import logger

load_dotenv()

class H2022Embedding(EmbeddingBase):
    """H2022 전용 임베딩 클래스"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        """
        H2022 임베딩 초기화
        
        Args:
            model_name: 사용할 OpenAI 임베딩 모델명
            dimension: 임베딩 차원 (기본값: 1536)
        
        Note:
            OpenAI API 키는 환경변수 OPENAI_API_KEY에서 자동으로 로드됩니다.
        """
        super().__init__(model_name=model_name, dimension=dimension)
        
        self.embedder = OpenAIEmbedder(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            dimension=dimension
        )
        
        logger.info(
            f"[H2022Embedding] H2022 임베딩 초기화 완료 "
            f"(model={model_name}, dimension={dimension})"
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
