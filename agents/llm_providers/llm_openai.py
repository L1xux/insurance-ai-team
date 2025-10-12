"""
OpenAI LLM 구현체
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
OpenAI API를 활용한 LLM 구현체입니다.
GPT 모델들을 사용하여 텍스트 생성, 코드 생성, 시각화 코드 생성 등의 기능을 제공하며,
비동기 처리를 지원하여 대용량 요청을 효율적으로 처리합니다.
"""
import os
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

from base.llm_base import LLMBase
from config.logging_config import logger

load_dotenv()


class LLMOpenAI(LLMBase):
    """OpenAI API 구현체"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name=model_name, provider="openai")
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY가 없습니다. Mock 모드")
            self.mock_mode = True
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """
        OpenAI API 사용 가능 여부 확인
        
        Returns:
            API 키가 설정되어 있고 클라이언트가 초기화된 경우 True
        """
        return not self.mock_mode and self.client is not None
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        OpenAI API를 통한 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            system_prompt: 시스템 프롬프트 (선택사항)
            temperature: 생성 온도 (0.0-2.0)
            max_tokens: 최대 토큰 수
            **kwargs: 추가 API 파라미터
            
        Returns:
            생성된 응답 텍스트
            
        Raises:
            Exception: API 호출 실패 시
        """
        if self.mock_mode:
            logger.warning("Mock 모드: 더미 응답")
            return f"[MOCK] {prompt[:50]}..."
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            raise
    
    async def close(self):
        """
        클라이언트 리소스 정리
        """
        if self.client:
            await self.client.close()