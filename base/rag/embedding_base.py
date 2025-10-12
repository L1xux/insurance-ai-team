"""
임베딩 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
텍스트를 벡터로 변환하는 임베딩 모델의 베이스 클래스입니다.
OpenAI, Ollama, Qwen 등 다양한 임베딩 모델을 통일된 인터페이스로 사용할 수 있도록 설계되었습니다.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from config.logging_config import logger


class EmbeddingBase(ABC):
    """임베딩 베이스 클래스 - 텍스트를 벡터로 변환"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        임베딩 모델 초기화
        
        Args:
            model_name: 사용할 임베딩 모델 이름
            dimension: 임베딩 벡터의 차원 수
        """
        self.model_name = model_name or self.__class__.__name__
        self.dimension = dimension
        self.embedding_count = 0
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트를 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 (리스트 형태)
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 벡터로 변환 (배치 처리)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        pass
    
    def get_dimension(self) -> int:
        """
        임베딩 벡터의 차원 수 반환
        
        Returns:
            벡터 차원 수
        """
        if self.dimension is None:
            logger.warning(f"[{self.model_name}] 차원 정보가 설정되지 않았습니다.")
            return 0
        return self.dimension
    
    def get_stats(self) -> Dict[str, Any]:
        """
        임베딩 통계 정보 반환
        
        Returns:
            모델명, 차원, 처리 횟수를 포함한 딕셔너리
        """
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'embedding_count': self.embedding_count
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            모델명과 차원 정보를 포함한 문자열
        """
        return f"{self.model_name}[dim={self.dimension}]"

