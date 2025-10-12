"""
OpenAI 임베딩 구현체
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
OpenAI의 임베딩 API를 사용하는 구현체입니다.
text-embedding-3-small, text-embedding-3-large 등의 모델을 지원합니다.
"""
import os
from typing import List, Optional
from openai import OpenAI

from base.rag.embedding_base import EmbeddingBase
from config.logging_config import logger


class OpenAIEmbedder(EmbeddingBase):
    """OpenAI 임베딩 구현체"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        OpenAI 임베딩 모델 초기화
        
        Args:
            model_name: 사용할 OpenAI 임베딩 모델명
            api_key: OpenAI API 키 (None이면 환경변수에서 로드)
            dimension: 임베딩 차원 (text-embedding-3 모델은 차원 축소 지원)
        """
        # 기본 차원 설정
        default_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if dimension is None:
            dimension = default_dimensions.get(model_name, 1536)
        
        super().__init__(model_name=model_name, dimension=dimension)
        
        # API 키 설정
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정하거나 api_key 인자를 전달하세요.")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"[{self.model_name}] OpenAI 임베더 초기화 완료 (dimension={self.dimension})")
    
    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트를 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 (리스트 형태)
        """
        try:
            # 빈 텍스트 처리
            if not text or not text.strip():
                logger.warning(f"[{self.model_name}] 빈 텍스트가 입력되었습니다. 0 벡터를 반환합니다.")
                return [0.0] * self.dimension
            
            # OpenAI API 호출
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimension if "text-embedding-3" in self.model_name else None
            )
            
            embedding = response.data[0].embedding
            self.embedding_count += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"[{self.model_name}] 임베딩 실패: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 벡터로 변환 (배치 처리)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            if not texts:
                logger.warning(f"[{self.model_name}] 빈 텍스트 리스트가 입력되었습니다.")
                return []
            
            # 빈 텍스트 필터링 및 인덱스 추적
            valid_texts = []
            valid_indices = []
            for idx, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(idx)
            
            if not valid_texts:
                logger.warning(f"[{self.model_name}] 모든 텍스트가 비어있습니다. 0 벡터를 반환합니다.")
                return [[0.0] * self.dimension for _ in texts]
            
            # OpenAI API 호출 (배치)
            response = self.client.embeddings.create(
                model=self.model_name,
                input=valid_texts,
                dimensions=self.dimension if "text-embedding-3" in self.model_name else None
            )
            
            # 결과를 원래 순서에 맞게 재구성
            embeddings = [[0.0] * self.dimension] * len(texts)
            for idx, valid_idx in enumerate(valid_indices):
                embeddings[valid_idx] = response.data[idx].embedding
            
            self.embedding_count += len(valid_texts)
            
            logger.info(f"[{self.model_name}] 배치 임베딩 완료: {len(valid_texts)}개 텍스트")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"[{self.model_name}] 배치 임베딩 실패: {str(e)}")
            raise

