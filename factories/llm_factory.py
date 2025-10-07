"""
LLM 팩토리 클래스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
다양한 LLM 제공자(OpenAI, Ollama 등)의 객체를 생성하는 팩토리 클래스입니다.
의존성 주입 패턴을 활용하여 LLM 구현체를 동적으로 생성하고,
제공자별 설정과 초기화를 관리합니다.
"""
from typing import Dict, Type, Optional
from enum import Enum

from base.llm_base import LLMBase

from llm.llm_openai import LLMOpenAI
from llm.llm_ollama import LLMOllama

from config.logging_config import logger


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMFactory:
    """LLM 인스턴스 생성 팩토리"""
    
    def __init__(self):
        self._providers: Dict[str, Type[LLMBase]] = {
            LLMProvider.OPENAI: LLMOpenAI,
            LLMProvider.OLLAMA: LLMOllama,
        }
        
        self._default_models = {
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.OLLAMA: "llama3",
        }
    
    def create_llm(
        self, 
        provider: str = "openai", 
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLMBase:
        """
        LLM 인스턴스 생성
        
        Args:
            provider: LLM 제공자 (openai, ollama)
            model_name: 사용할 모델명 (None이면 기본값 사용)
            **kwargs: LLM 초기화에 전달할 추가 인자
            
        Returns:
            생성된 LLM 인스턴스
            
        Raises:
            ValueError: 알 수 없는 제공자인 경우
        """
        provider = provider.lower()
        
        llm_class = self._providers.get(provider)
        if not llm_class:
            available = ', '.join(self._providers.keys())
            raise ValueError(f"알 수 없는 제공자: {provider}\n가능: {available}")
        
        if model_name is None:
            model_name = self._default_models.get(provider)
        
        llm = llm_class(model_name=model_name, **kwargs)
        
        if not llm.is_available():
            logger.warning(f"{provider} 사용 불가")
        
        logger.info(f"LLM 생성: {provider}/{model_name}")
        return llm
    