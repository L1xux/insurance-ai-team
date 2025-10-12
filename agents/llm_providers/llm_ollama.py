"""
Ollama LLM 구현체
=========================
Author: Jin
Date: 2025.09.17
Version: 3.0

Description:
Ollama 로컬 LLM을 활용한 구현체입니다.
"""
from typing import Any
import httpx

from langchain_ollama import ChatOllama

from base.agent.llm_base import LLMBase
from config.logging_config import logger


class LLMOllama(LLMBase):
    """Ollama 로컬 LLM 구현체"""
    
    def __init__(
        self, 
        model_name: str = "llama3.2", 
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7
    ):
        """
        LLMOllama 초기화
        
        Args:
            model_name: Ollama 모델 이름 (llama3.2, mistral 등)
            base_url: Ollama 서버 URL
            temperature: 생성 온도 (0.0-2.0)
        """
        super().__init__(model_name=model_name, provider="ollama")
        self.base_url = base_url
        self.temperature = temperature
        
        # ChatOllama 모델 생성
        self.chat_model = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )
        logger.info(f"[LLMOllama] ChatOllama 생성: {model_name} (temp={temperature})")
    
    def get_model(self) -> Any:
        """
        ChatOllama 모델 반환
        
        Returns:
            ChatOllama 인스턴스
        """
        return self.chat_model
    
    def is_available(self) -> bool:
        """
        Ollama 서버 사용 가능 여부 확인
        
        Returns:
            Ollama 서버가 실행 중이고 접근 가능한 경우 True
        """
        try:
            response = httpx.get(f"{self.base_url}/api/version", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False