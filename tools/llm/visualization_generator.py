"""
LLM 시각화 코드 생성기
=========================
Author: Jin
Date: 2025.09.17
Version: 1.1 (의존성 주입 패턴 적용)

Description:
LLM을 활용하여 데이터 분석 결과를 바탕으로 Python 시각화 코드를 자동 생성하는 클래스입니다.
외부에서 주입된 LLM 객체를 사용하여 사용자 지시문에 따른 맞춤형 시각화 코드를 생성하며,
생성된 코드의 안전성과 실행 가능성을 보장하는 후처리 기능을 포함합니다.
"""
from typing import Dict, Any, Optional, List

from config.logging_config import logger


class VisualizationGenerator:
    """데이터 분석 결과를 바탕으로 시각화 코드를 생성하는 클래스"""
    
    def __init__(self, llm):
        """
        Args:
            llm: LLM 인터페이스 객체 (LLMFactory로 생성된 객체)
                 generate_response, build_context, extract_available_fields 메서드 필요
        """
        self.llm = llm
        self.system_prompt = self._create_system_prompt()
        logger.info(f"VisualizationGenerator 초기화: LLM={type(llm).__name__}")
    
    def _create_system_prompt(self) -> str:
        """
        시각화 코드 생성을 위한 시스템 프롬프트
        
        Returns:
            시스템 프롬프트 문자열
        """
        return """당신은 데이터 시각화 전문가입니다. 주어진 데이터 분석 결과를 바탕으로 Python 시각화 코드를 생성합니다.

**핵심 원칙:**
1. **사용자 요청 최우선**: 사용자가 원하는 시각화를 반드시 구현하세요
2. **데이터 컬럼 확인**: 반드시 사용 가능한 컬럼을 확인하고 존재하는 컬럼만 사용하세요
3. **대체 컬럼 사용**: 요청한 컬럼이 없으면 의미상 가장 비슷한 컬럼을 찾아 사용하세요

**기술적 규칙:**
4. 함수 형태로 작성 (create_visualization(df) 함수)
5. matplotlib, seaborn, pandas만 사용
6. 실행 가능한 완전한 코드 생성
7. 적절한 제목, 축 레이블, 범례 포함
8. 컬러 팔레트와 스타일 적용 (plt.style.use('seaborn-v0_8') 또는 'default' 사용)
9. 여러 차트를 subplot으로 구성
10. 코드만 반환, 설명문 제외

**시각화 우선순위:**
1. 카테고리별 분석 (막대 그래프)
2. 수치형 데이터 분포 (히스토그램, 박스플롯)
3. 상관관계 분석 (산점도, 히트맵)
4. 트렌드 분석 (선 그래프)
5. 비교 분석 (그룹별 비교)

**코드 형식 예시:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_visualization(df):
    plt.style.use('seaborn-v0_8')
    
    # 반드시 사용 가능한 컬럼 확인
    available_columns = df.columns.tolist()
    print(f"사용 가능한 컬럼: {available_columns}")
    
    # 사용 가능한 컬럼만 사용하여 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 차트 1, 2, 3, 4...
    
    plt.tight_layout()
    return fig
```

**중요:**
- 사용자가 구체적인 요청을 하지 않으면, 데이터의 특성을 분석하여 가장 유의미한 시각화를 자율적으로 선택하세요.
- 컬럼명은 정확히 확인하고 사용하세요. 존재하지 않는 컬럼을 사용하면 오류가 발생합니다."""

    async def generate_visualization_code(
        self, 
        analysis_results: Dict[str, Any], 
        user_context: Optional[str] = None,
        specific_fields: Optional[List[str]] = None
    ) -> str:
        """
        분석 결과를 바탕으로 시각화 코드 생성
        
        Args:
            analysis_results: 데이터 분석 결과
            user_context: 사용자가 제공한 컨텍스트 (예: "카테고리별 평점 분석")
            specific_fields: 특정 필드들 지정 (옵션)
            
        Returns:
            str: 생성된 시각화 코드
        """
        try:
            logger.info("시각화 코드 생성 시작")
            
            # 컨텍스트 구성 (LLM의 build_context 메서드 사용)
            context = self.llm.build_context(analysis_results, user_context)
            
            # 사용 가능한 필드 추출 (LLM의 extract_available_fields 메서드 사용)
            available_fields = self.llm.extract_available_fields(analysis_results)
            
            # 프롬프트 구성
            prompt = self._build_prompt(context, available_fields, specific_fields, user_context)
            
            # LLM을 통한 코드 생성
            generated_code = await self.llm.generate_response(prompt, self.system_prompt)
            
            # 코드 후처리
            cleaned_code = self._clean_generated_code(generated_code)
            
            logger.info("시각화 코드 생성 완료")
            return cleaned_code
            
        except Exception as e:
            logger.error(f"시각화 코드 생성 실패: {str(e)}")
            raise e
    
    def _build_prompt(
        self, 
        context: str, 
        available_fields: List[str], 
        specific_fields: Optional[List[str]] = None,
        user_context: Optional[str] = None
    ) -> str:
        """
        시각화 코드 생성을 위한 프롬프트 구성
        
        Args:
            context: 데이터 분석 컨텍스트
            available_fields: 사용 가능한 필드 목록
            specific_fields: 특정 필드 목록 (선택사항)
            user_context: 사용자 컨텍스트 (선택사항)
            
        Returns:
            구성된 프롬프트 문자열
        """
        prompt_parts = [
            "다음 데이터 분석 결과를 바탕으로 시각화 코드를 생성해주세요:",
            "",
            "**데이터 정보:**",
            context,
            "",
            f"**📊 사용 가능한 데이터 필드:** {', '.join(available_fields)}",
        ]
        
        if specific_fields:
            prompt_parts.extend([
                "",
                f"**집중할 필드:** {', '.join(specific_fields)}"
            ])
        
        if user_context:
            prompt_parts.extend([
                "",
                f"**🎯 사용자 요청 (최우선):** {user_context}",
                "",
                "**중요 지침:**",
                "1. 사용자 요청을 최대한 충족하는 시각화를 생성하세요",
                "2. 요청한 컬럼이 없다면 가장 비슷한 의미의 컬럼을 찾아서 사용하세요",
                "3. 대체 컬럼을 사용할 때는 제목이나 주석에서 명확히 설명하세요",
            ])
        else:
            prompt_parts.extend([
                "",
                "**자율 분석 요청:** 데이터의 특성을 파악하여 가장 유의미한 시각화를 생성해주세요.",
                "필드 간의 연관성, 분포, 패턴 등을 고려하여 인사이트를 도출할 수 있는 시각화를 만들어주세요."
            ])
        
        prompt_parts.extend([
            "",
            "실행 가능한 완전한 Python 코드만 생성해주세요."
        ])
        
        return "\n".join(prompt_parts)
    
    def _clean_generated_code(self, generated_code: str) -> str:
        """
        생성된 코드 정리 및 검증
        
        Args:
            generated_code: 정리할 생성된 코드
            
        Returns:
            정리된 코드 문자열
        """
        # 코드 블록 마커 제거
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0]
        
        # 앞뒤 공백 제거
        generated_code = generated_code.strip()
        
        # 기본 import 확인 및 추가
        required_imports = [
            "import matplotlib.pyplot as plt",
            "import seaborn as sns", 
            "import pandas as pd"
        ]
        
        for import_stmt in required_imports:
            if import_stmt not in generated_code:
                generated_code = import_stmt + "\n" + generated_code
        
        # 오래된 matplotlib 스타일명을 올바른 스타일명으로 수정
        style_replacements = {
            "'seaborn-darkgrid'": "'seaborn-v0_8'",
            '"seaborn-darkgrid"': '"seaborn-v0_8"',
            "'seaborn-whitegrid'": "'seaborn-v0_8'",
            '"seaborn-whitegrid"': '"seaborn-v0_8"',
            "'seaborn-dark'": "'seaborn-v0_8'",
            '"seaborn-dark"': '"seaborn-v0_8"',
            "'seaborn-white'": "'seaborn-v0_8'",
            '"seaborn-white"': '"seaborn-v0_8"',
            "'seaborn-ticks'": "'seaborn-v0_8'",
            '"seaborn-ticks"': '"seaborn-v0_8"'
        }
        
        for old_style, new_style in style_replacements.items():
            generated_code = generated_code.replace(old_style, new_style)
        
        return generated_code
    
    async def close(self) -> None:
        """
        리소스 정리
        """
        if hasattr(self.llm, 'close'):
            await self.llm.close()
        logger.info("VisualizationGenerator 리소스 정리 완료")