"""
LLM 서비스 베이스 클래스
=========================
Author: Jin
Date: 2025.09.30
Version: 2.0

Description:
모든 LLM 구현체의 기본이 되는 추상 베이스 클래스입니다.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from config.logging_config import logger


class LLMBase(ABC):
    """LLM 서비스를 위한 기본 추상 클래스"""
    
    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider
        self.mock_mode = False
        logger.info(f"LLM 초기화: {provider}/{model_name}")
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        LLM 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            system_prompt: 시스템 프롬프트 (선택사항)
            **kwargs: 추가 키워드 인자
            
        Returns:
            LLM 생성 응답 텍스트
        """
        pass
    
    @abstractmethod
    async def close(self):
        """
        리소스 정리
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        사용 가능 여부 확인
        
        Returns:
            LLM 사용 가능 여부
        """
        pass
    
    def build_context(
        self, 
        analysis_results: Dict[str, Any], 
        user_context: Optional[str] = None
    ) -> str:
        """
        분석 결과 기반 컨텍스트 구성
        
        Args:
            analysis_results: 분석 결과 딕셔너리
            user_context: 사용자 추가 컨텍스트 (선택사항)
            
        Returns:
            구성된 컨텍스트 문자열
        """
        context_parts = []
        
        if 'basic_statistics' in analysis_results:
            basic_stats = analysis_results['basic_statistics']
            if 'shape' in basic_stats:
                context_parts.append(
                    f"데이터 크기: {basic_stats['shape'][0]:,}행 {basic_stats['shape'][1]}열"
                )
            if 'columns' in basic_stats:
                context_parts.append(f"컬럼: {', '.join(basic_stats['columns'])}")
        
        if 'domain_analysis' in analysis_results:
            domain_analysis = analysis_results['domain_analysis']
            for analysis_type, analysis_data in domain_analysis.items():
                summary = self._summarize_analysis_data(analysis_type, analysis_data)
                if summary:
                    context_parts.append(summary)
        
        if user_context:
            context_parts.append(f"사용자 요청: {user_context}")
        
        return "\n".join(context_parts)
    
    def _summarize_analysis_data(self, analysis_type: str, analysis_data: Any) -> str:
        """
        분석 데이터 요약
        
        Args:
            analysis_type: 분석 타입
            analysis_data: 분석 데이터
            
        Returns:
            요약된 분석 데이터 문자열
        """
        if not isinstance(analysis_data, dict):
            return f"{analysis_type}: {str(analysis_data)}"
        
        summary_parts = []
        
        if 'statistics' in analysis_data:
            stats = analysis_data['statistics']
            if isinstance(stats, dict):
                if 'mean' in stats:
                    summary_parts.append(f"평균: {stats['mean']:.2f}")
                if 'count' in stats or 'total' in stats:
                    count = stats.get('count', stats.get('total'))
                    summary_parts.append(f"총 개수: {count}")
        
        if 'distribution' in analysis_data:
            dist = analysis_data['distribution']
            if isinstance(dist, dict) and len(dist) <= 5:
                dist_str = ', '.join([f"{k}: {v}" for k, v in list(dist.items())[:3]])
                summary_parts.append(f"분포: {dist_str}")
        
        if summary_parts:
            return f"{analysis_type}: {', '.join(summary_parts)}"
        
        return f"{analysis_type}: 분석 완료"
    
    def extract_available_fields(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        시각화 가능 필드 추출
        
        Args:
            analysis_results: 분석 결과 딕셔너리
            
        Returns:
            시각화 가능한 필드 목록
        """
        fields = []
        
        if 'basic_statistics' in analysis_results:
            basic_stats = analysis_results['basic_statistics']
            if 'columns' in basic_stats:
                fields.extend(basic_stats['columns'])
        
        return fields
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            클래스명, 프로바이더, 모델명을 포함한 문자열
        """
        return f"{self.__class__.__name__}[{self.provider}/{self.model_name}]"