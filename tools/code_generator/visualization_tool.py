"""
Visualization Tool
=========================
Author: Jin
Date: 2025.10.13
Version: 2.0

Description:
데이터 시각화를 생성하는 BaseTool입니다.
사용자 요청을 받아 LLM으로 시각화 코드를 생성하고 안전하게 실행합니다.
"""
from typing import Optional, Dict, Any
import json
import pandas as pd
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.logging_config import logger
from utils.code_processor import CodeProcessor
from models.tool_model import ToolSchema

class VisualizationTool(BaseTool):
    """데이터 시각화 생성 Tool"""
    
    name: str = "visualization_generator"
    description: str = """데이터 시각화를 생성하는 도구입니다.
    
    사용자 요청에 따라 Python 시각화 코드를 생성하고 실행하여 이미지를 저장합니다.
    
    입력 형식: {
        "request": "사용자 시각화 요청",
        "data_summary": "데이터 요약 정보 (optional)",
        "available_columns": ["컬럼1", "컬럼2", ...] (optional)
    }
    
    출력: 저장된 이미지 파일 경로 및 실행 결과
    """
    
    llm: Any = None
    code_processor: Any = None
    dataframe: Any = None
    code_generation_prompt: Any = None
    
    def __init__(
        self, 
        llm_model: str = "gpt-4o-mini",
        output_dir: str = "data",
        dataframe: Optional[pd.DataFrame] = None
    ):
        """
        Tool 초기화
        
        Args:
            llm_model: 코드 생성용 LLM 모델
            output_dir: 이미지 저장 디렉토리
            dataframe: 시각화할 DataFrame
        """
        super().__init__()
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.code_processor = CodeProcessor(output_dir)
        self.dataframe = dataframe
        self._setup_prompt()
        logger.info(f"[{self.name}] 초기화 완료")
    
    def _setup_prompt(self) -> None:
        """시각화 코드 생성 프롬프트 설정"""
        self.code_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 데이터 시각화 전문가입니다.

**핵심 원칙:**
1. 사용자 요청을 최우선으로 구현하세요
2. 사용 가능한 컬럼만 사용하세요
3. matplotlib, seaborn, pandas만 사용하세요

**코드 형식:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_visualization(df):
    plt.style.use('seaborn-v0_8')
    
    # 사용 가능한 컬럼 확인
    available_columns = df.columns.tolist()
    print(f"사용 가능한 컬럼: {{available_columns}}")
    
    # 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 차트들 생성...
    
    plt.tight_layout()
    return fig
```

**중요:**
- 반드시 create_visualization(df) 함수로 작성
- 코드만 반환, 설명문 제외
- 적절한 제목, 축 레이블, 범례 포함"""),
            ("user", """사용자 요청: {request}

{context}

위 정보를 바탕으로 시각화 코드를 생성하세요.""")
        ])
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """DataFrame 설정"""
        self.dataframe = df
        logger.info(f"[{self.name}] DataFrame 설정: {df.shape}")
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        시각화 생성 실행
        
        Args:
            query: JSON 문자열 (request, data_summary, available_columns)
            run_manager: callback manager
            
        Returns:
            실행 결과 (JSON 문자열)
        """
        try:
            # JSON 파싱
            try:
                params = json.loads(query)
                request = params.get('request', query)
                data_summary = params.get('data_summary', '')
                available_columns = params.get('available_columns', [])
            except (json.JSONDecodeError, TypeError):
                request = query
                data_summary = ''
                available_columns = []
            
            logger.info(f"[{self.name}] 시각화 요청: {request[:50]}...")
            
            # DataFrame 체크
            if self.dataframe is None:
                return json.dumps({
                    'success': False,
                    'error': 'DataFrame이 설정되지 않았습니다.'
                }, ensure_ascii=False)
            
            # 사용 가능한 컬럼 자동 추출
            if not available_columns:
                available_columns = self.dataframe.columns.tolist()
            
            # Context 구성
            context_parts = []
            if data_summary:
                context_parts.append(f"데이터 요약:\n{data_summary}")
            context_parts.append(f"사용 가능한 컬럼:\n{', '.join(available_columns)}")
            context_parts.append(f"데이터 형태: {self.dataframe.shape[0]}행 × {self.dataframe.shape[1]}열")
            context = "\n\n".join(context_parts)
            
            # 1. LLM으로 시각화 코드 생성
            code_generation_chain = self.code_generation_prompt | self.llm
            response = code_generation_chain.invoke({
                "request": request,
                "context": context
            })
            
            generated_code = response.content.strip()
            
            # 코드 추출 (```python ... ``` 제거)
            if '```python' in generated_code:
                generated_code = generated_code.split('```python')[1].split('```')[0].strip()
            elif '```' in generated_code:
                generated_code = generated_code.split('```')[1].split('```')[0].strip()
            
            logger.info(f"[{self.name}] 코드 생성 완료: {len(generated_code)} 문자")
            
            # 2. 코드 실행
            success, output_path, error_message = self.code_processor.execute_visualization_code(
                code=generated_code,
                df=self.dataframe,
                save_plot=True,
                timeout=30
            )
            
            if success:
                return json.dumps({
                    'success': True,
                    'output_path': str(output_path),
                    'message': f'시각화 생성 완료: {output_path}',
                    'generated_code': generated_code
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    'success': False,
                    'error': error_message,
                    'generated_code': generated_code
                }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[{self.name}] 실행 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="데이터 시각화 생성 - 사용자 요청에 따라 Python 시각화 코드를 생성하고 실행",
            parameters={
                'request': {
                    'type': 'string',
                    'description': '시각화 요청 (예: "카테고리별 평점 분포를 보여줘")'
                },
                'data_summary': {
                    'type': 'string',
                    'description': '데이터 요약 정보 (optional)'
                },
                'available_columns': {
                    'type': 'array',
                    'description': '사용 가능한 컬럼 목록 (optional)'
                }
            },
            required_params=['request']
        )

