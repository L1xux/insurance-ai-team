"""
LLM 서비스 베이스 클래스
=========================
Author: Jin
Date: 2025.09.30
Version: 3.0

Description:
모든 LLM 구현체의 기본이 되는 추상 베이스 클래스입니다.
"""
from abc import ABC, abstractmethod
from typing import Any

from config.logging_config import logger

class LLMBase(ABC):
    """LLM 서비스를 위한 기본 추상 클래스"""
    
    def __init__(self, model_name: str, provider: str):
        """
        LLM 초기화
        
        Args:
            model_name: 모델 이름
            provider: 제공자 (openai, ollama 등)
        """
        self.model_name = model_name
        self.provider = provider
        self.mock_mode = False
        logger.info(f"LLM 초기화: {provider}/{model_name}")
    
    @abstractmethod
    def get_model(self) -> Any:
        """
        LLM 모델 반환 (Agent에서 사용)
        
        Returns:
            ChatModel 인스턴스 (ChatOpenAI, ChatOllama 등)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        LLM 사용 가능 여부 확인
        
        Returns:
            LLM 사용 가능 여부
        """
        pass
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            클래스명, 프로바이더, 모델명을 포함한 문자열
        """
        return f"{self.__class__.__name__}[{self.provider}/{self.model_name}]"