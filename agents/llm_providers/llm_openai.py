"""
OpenAI LLM 구현체
=========================
Author: Jin
Date: 2025.09.17
Version: 3.0

Description:
OpenAI API를 활용한 LLM 구현체입니다.
"""
import os
from typing import Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from base.agent.llm_base import LLMBase
from config.logging_config import logger

load_dotenv()


class LLMOpenAI(LLMBase):
    """OpenAI LLM 구현체"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        """
        LLMOpenAI 초기화
        
        Args:
            model_name: OpenAI 모델 이름 (gpt-4o, gpt-4o-mini 등)
            temperature: 생성 온도 (0.0-2.0)
        """
        super().__init__(model_name=model_name, provider="openai")
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        
        if self.api_key:
            # 모델 생성
            self.chat_model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=self.api_key
            )
            logger.info(f"[LLMOpenAI] ChatOpenAI 생성: {model_name} (temp={temperature})")

        else:
            logger.warning("OPENAI_API_KEY가 없습니다. Mock 모드")
            self.mock_mode = True
            self.chat_model = None
    
    def get_model(self) -> Any:
        """
        ChatOpenAI 모델 반환
        
        Returns:
            ChatOpenAI 인스턴스
        """
        if self.mock_mode:
            logger.warning("[LLMOpenAI] Mock 모드: ChatOpenAI 모델 없음")
            return None
        return self.chat_model
    
    def is_available(self) -> bool:
        """
        OpenAI API 사용 가능 여부 확인
        
        Returns:
            API 키가 설정되어 있고 클라이언트가 초기화된 경우 True
        """
        return not self.mock_mode and self.chat_model is not None