"""
Ollama LLM 구현체
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
Ollama 로컬 LLM을 활용한 구현체입니다.
로컬에서 실행되는 다양한 오픈소스 모델들(Llama, Mistral 등)을 사용하여
텍스트 생성, 코드 생성, 시각화 코드 생성 등의 기능을 제공하며,
API 키 없이도 사용할 수 있는 로컬 LLM 솔루션을 제공합니다.
"""
from typing import Optional
import httpx

from base.llm_base import LLMBase
from config.logging_config import logger


class LLMOllama(LLMBase):
    """Ollama 로컬 LLM 구현체"""
    
    def __init__(
        self, 
        model_name: str = "llama3.2", 
        base_url: str = "http://localhost:11434"
    ):
        super().__init__(model_name=model_name, provider="ollama")
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
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
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Ollama API를 통한 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            system_prompt: 시스템 프롬프트 (선택사항)
            temperature: 생성 온도 (0.0-2.0)
            **kwargs: 추가 API 파라미터
            
        Returns:
            생성된 응답 텍스트
            
        Raises:
            Exception: API 호출 실패 시
        """
        try:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            payload = {
                "model": self.model_name,
                "messages": messages, 
                "stream": False,
                "options": {"temperature": temperature}
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/chat",  
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Ollama API 호출 실패: {e}")
            raise
    
    async def close(self):
        """
        HTTP 클라이언트 리소스 정리
        """
        await self.client.aclose()