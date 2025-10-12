"""
Demographic Analysis Tool
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
인구통계 분석 Tool (Customer Insight Agent용)
"""
from typing import Dict, Any, Optional
import pandas as pd
import json

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from utils.data.insurance_2022_data_analyzer import Insurance2022DataAnalyzer
from config.logging_config import logger
from models.tool_model import ToolSchema


class H2022DemographicTool(BaseTool):
    """H2022 인구통계 분석 Tool"""
    
    name: str = "h2022_demographic_analysis"
    description: str = """H2022 인구통계 분석 도구입니다.
    
    분석 유형:
    - age_distribution: 연령 분포 분석
    - gender_distribution: 성별 분포 분석
    - insurance_coverage: 보험 가입 현황 분석
    - demographic_disparities: 인구통계학적 격차 분석
    
    입력 형식: {"analysis_type": "age_distribution"}
    """
    
    analyzer: Any = None
    
    def __init__(self, analyzer: Insurance2022DataAnalyzer):
        """
        Tool 초기화
        
        Args:
            analyzer: 이미 로드된 Insurance2022DataAnalyzer 인스턴스
        """
        super().__init__()
        self.analyzer = analyzer
        logger.info(f"[{self.name}] 초기화 완료 (데이터: {len(analyzer.df)}개)")
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        인구통계 분석 실행
        
        Args:
            query: JSON 문자열 또는 analysis_type
            run_manager: callback manager
            
        Returns:
            분석 결과 (JSON 문자열)
        """
        try:
            # JSON 파싱 시도
            try:
                params = json.loads(query)
                analysis_type = params.get('analysis_type', 'age_distribution')
            except (json.JSONDecodeError, TypeError):
                # 단순 문자열인 경우
                analysis_type = query if query else 'age_distribution'
            
            logger.info(f"[{self.name}] 실행: {analysis_type}")
            
            if analysis_type == 'age_distribution':
                result = self._analyze_age_distribution()
            elif analysis_type == 'gender_distribution':
                result = self._analyze_gender_distribution()
            elif analysis_type == 'insurance_coverage':
                result = self.analyzer._analyze_insurance_coverage()
            elif analysis_type == 'demographic_disparities':
                result = self.analyzer._analyze_demographic_disparities()
            else:
                return json.dumps({
                    'success': False,
                    'error': f"지원하지 않는 분석 유형: {analysis_type}"
                }, ensure_ascii=False)
            
            return json.dumps({
                'success': True,
                'data': result
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[{self.name}] 실행 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
    
    def _analyze_age_distribution(self) -> Dict[str, Any]:
        """연령 분포 분석"""
        df = self.analyzer.df
        
        if 'age_group' not in df:
            return {'error': 'age_group 컬럼이 없습니다.'}
        
        return {
            'age_group_counts': df['age_group'].value_counts().to_dict(),
            'age_group_percentages': (df['age_group'].value_counts(normalize=True) * 100).to_dict()
        }
    
    def _analyze_gender_distribution(self) -> Dict[str, Any]:
        """성별 분포 분석"""
        df = self.analyzer.df
        
        if 'SEX' not in df:
            return {'error': 'SEX 컬럼이 없습니다.'}
        
        sex_map = {1: 'Male', 2: 'Female'}
        gender_counts = df['SEX'].map(sex_map).value_counts().to_dict()
        
        return {
            'gender_counts': gender_counts,
            'gender_percentages': {k: (v / len(df)) * 100 for k, v in gender_counts.items()}
        }
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="인구통계 분석 - 연령, 성별, 보험 가입 현황, 인구통계학적 격차 분석",
            parameters={
                'analysis_type': {
                    'type': 'string',
                    'description': '분석 유형',
                    'enum': ['age_distribution', 'gender_distribution', 'insurance_coverage', 'demographic_disparities']
                }
            },
            required_params=['analysis_type']
        )

