"""
Ollama 임베딩 구현체
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
로컬 Ollama 서버의 임베딩 모델을 사용하는 구현체입니다.
nomic-embed-text, mxbai-embed-large 등의 모델을 지원합니다.
"""
import requests
from typing import List, Optional

from base.rag.embedding_base import EmbeddingBase
from config.logging_config import logger


class OllamaEmbedder(EmbeddingBase):
    """Ollama 임베딩 구현체"""
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: Optional[int] = None
    ):
        """
        Ollama 임베딩 모델 초기화
        
        Args:
            model_name: 사용할 Ollama 임베딩 모델명
            base_url: Ollama 서버 URL
            dimension: 임베딩 차원 (None이면 자동 감지)
        """
        # 기본 차원 설정
        default_dimensions = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384
        }
        
        if dimension is None:
            dimension = default_dimensions.get(model_name, 768)
        
        super().__init__(model_name=model_name, dimension=dimension)
        
        self.base_url = base_url.rstrip('/')
        self.embed_url = f"{self.base_url}/api/embeddings"
        
        # 서버 연결 확인
        self._check_connection()
        
        logger.info(f"[{self.model_name}] Ollama 임베더 초기화 완료 (dimension={self.dimension})")
    
    def _check_connection(self) -> None:
        """
        Ollama 서버 연결 확인
        
        Raises:
            ConnectionError: 서버 연결 실패 시
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"[{self.model_name}] Ollama 서버 연결 확인")
        except Exception as e:
            logger.error(f"[{self.model_name}] Ollama 서버 연결 실패: {str(e)}")
            raise ConnectionError(f"Ollama 서버에 연결할 수 없습니다: {self.base_url}")
    
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
            
            # Ollama API 호출
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(
                self.embed_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            # 차원 확인 및 업데이트
            if len(embedding) != self.dimension:
                logger.warning(
                    f"[{self.model_name}] 예상 차원({self.dimension})과 "
                    f"실제 차원({len(embedding)})이 다릅니다. 차원을 업데이트합니다."
                )
                self.dimension = len(embedding)
            
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
            
            # Ollama는 배치 API를 지원하지 않으므로 순차 처리
            embeddings = []
            valid_count = 0
            
            for text in texts:
                if text and text.strip():
                    embedding = self.embed_text(text)
                    embeddings.append(embedding)
                    valid_count += 1
                else:
                    # 빈 텍스트는 0 벡터로
                    embeddings.append([0.0] * self.dimension)
            
            logger.info(f"[{self.model_name}] 배치 임베딩 완료: {valid_count}개 텍스트")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"[{self.model_name}] 배치 임베딩 실패: {str(e)}")
            raise

