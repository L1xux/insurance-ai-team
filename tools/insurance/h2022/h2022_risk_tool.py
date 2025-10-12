"""
H2022 Risk Analysis Tool
=========================
Author: Jin
Date: 2025.10.13
Version: 1.0

Description:
리스크 분석 Tool (고위험군 식별, 리스크 스코어링, 이상치 탐지)
"""
from typing import Dict, Any, Optional
import pandas as pd
import json
import numpy as np

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from utils.data.insurance_2022_data_analyzer import Insurance2022DataAnalyzer
from config.logging_config import logger
from models.tool_model import ToolSchema


class H2022RiskTool(BaseTool):
    """리스크 분석 Tool"""
    
    name: str = "h2022_risk_analysis"
    description: str = """리스크 분석 도구입니다.
    
    분석 유형:
    - high_risk_segments: 고위험 세그먼트 식별
    - risk_scoring: 리스크 스코어링
    - outlier_detection: 이상치 탐지
    - risk_factors: 리스크 요인 분석
    
    입력 형식: {"analysis_type": "high_risk_segments"}
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
        리스크 분석 실행
        
        Args:
            query: JSON 문자열 또는 analysis_type
            run_manager:  callback manager
            
        Returns:
            분석 결과 (JSON 문자열)
        """
        try:
            # JSON 파싱
            try:
                params = json.loads(query)
                analysis_type = params.get('analysis_type', 'high_risk_segments')
                threshold = params.get('threshold', 10000)
            except (json.JSONDecodeError, TypeError):
                analysis_type = query if query else 'high_risk_segments'
                threshold = 10000
            
            logger.info(f"[{self.name}] 실행: {analysis_type}")
            
            if analysis_type == 'high_risk_segments':
                result = self._identify_high_risk_segments()
            elif analysis_type == 'risk_scoring':
                result = self._calculate_risk_scores()
            elif analysis_type == 'outlier_detection':
                result = self._detect_outliers(threshold)
            elif analysis_type == 'risk_factors':
                result = self._analyze_risk_factors()
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
    
    def _identify_high_risk_segments(self) -> Dict[str, Any]:
        """고위험 세그먼트 식별"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        result = {}
        
        # 연령대별 고위험군
        if 'age_group' in df.columns:
            age_risk = df.groupby('age_group')['TOTEXP22'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
            result['by_age'] = {
                str(idx): {
                    'mean_expenditure': float(row['mean']),
                    'median_expenditure': float(row['median']),
                    'count': int(row['count']),
                    'risk_level': 'High' if row['mean'] > df['TOTEXP22'].mean() * 1.5 else 'Medium' if row['mean'] > df['TOTEXP22'].mean() else 'Low'
                }
                for idx, row in age_risk.iterrows()
            }
        
        # 보험 유형별 고위험군
        if 'insurance_type' in df.columns:
            insurance_risk = df.groupby('insurance_type')['TOTEXP22'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
            result['by_insurance'] = {
                str(idx): {
                    'mean_expenditure': float(row['mean']),
                    'median_expenditure': float(row['median']),
                    'count': int(row['count'])
                }
                for idx, row in insurance_risk.iterrows()
            }
        
        # 건강 상태별 고위험군
        if 'poor_health' in df.columns:
            poor_health_mean = df[df['poor_health'] == 1]['TOTEXP22'].mean()
            good_health_mean = df[df['poor_health'] == 0]['TOTEXP22'].mean()
            
            result['by_health_status'] = {
                'poor_health': {
                    'mean_expenditure': float(poor_health_mean),
                    'count': int((df['poor_health'] == 1).sum())
                },
                'good_health': {
                    'mean_expenditure': float(good_health_mean),
                    'count': int((df['poor_health'] == 0).sum())
                },
                'risk_ratio': float(poor_health_mean / good_health_mean) if good_health_mean > 0 else 0
            }
        
        return result
    
    def _calculate_risk_scores(self) -> Dict[str, Any]:
        """리스크 스코어링"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        # 간단한 리스크 스코어 계산 (0-100)
        # 요소: 나이, 건강상태, 과거 지출
        risk_scores = []
        
        # 표준화된 지출 (0-1)
        max_exp = df['TOTEXP22'].max()
        if max_exp > 0:
            normalized_exp = df['TOTEXP22'] / max_exp
        else:
            normalized_exp = pd.Series(0, index=df.index)
        
        # 나이 점수 (노인일수록 높음)
        if 'AGE22X' in df.columns:
            age_score = df['AGE22X'] / 100
        else:
            age_score = pd.Series(0.5, index=df.index)
        
        # 건강 상태 점수
        if 'poor_health' in df.columns:
            health_score = df['poor_health'] * 0.3
        else:
            health_score = pd.Series(0, index=df.index)
        
        # 최종 리스크 스코어 (0-100)
        risk_score = (normalized_exp * 0.5 + age_score * 0.3 + health_score) * 100
        
        # 리스크 등급 분류
        risk_categories = pd.cut(risk_score, bins=[0, 25, 50, 75, 100], 
                                 labels=['Low', 'Medium', 'High', 'Very High'])
        
        result = {
            'score_distribution': {
                'mean': float(risk_score.mean()),
                'median': float(risk_score.median()),
                'std': float(risk_score.std())
            },
            'risk_categories': risk_categories.value_counts().to_dict(),
            'high_risk_count': int((risk_score > 75).sum()),
            'high_risk_percentage': float((risk_score > 75).sum() / len(df) * 100)
        }
        
        return result
    
    def _detect_outliers(self, threshold: float = 10000) -> Dict[str, Any]:
        """이상치 탐지"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        # IQR 방법으로 이상치 탐지
        Q1 = df['TOTEXP22'].quantile(0.25)
        Q3 = df['TOTEXP22'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['TOTEXP22'] < lower_bound) | (df['TOTEXP22'] > upper_bound)]
        high_outliers = df[df['TOTEXP22'] > upper_bound]
        
        result = {
            'outlier_detection': {
                'total_outliers': len(outliers),
                'high_outliers': len(high_outliers),
                'outlier_percentage': float(len(outliers) / len(df) * 100),
                'iqr': float(IQR),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            },
            'extreme_cases': {
                'above_threshold': int((df['TOTEXP22'] > threshold).sum()),
                'max_expenditure': float(df['TOTEXP22'].max()),
                'top_10_percentile': float(df['TOTEXP22'].quantile(0.9))
            }
        }
        
        return result
    
    def _analyze_risk_factors(self) -> Dict[str, Any]:
        """리스크 요인 분석"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        result = {
            'key_factors': {}
        }
        
        # 나이 요인
        if 'is_senior' in df.columns:
            senior_mean = df[df['is_senior'] == 1]['TOTEXP22'].mean()
            non_senior_mean = df[df['is_senior'] == 0]['TOTEXP22'].mean()
            result['key_factors']['age_senior'] = {
                'senior_avg': float(senior_mean),
                'non_senior_avg': float(non_senior_mean),
                'risk_multiplier': float(senior_mean / non_senior_mean) if non_senior_mean > 0 else 0
            }
        
        # 건강 상태 요인
        if 'poor_health' in df.columns:
            poor_mean = df[df['poor_health'] == 1]['TOTEXP22'].mean()
            good_mean = df[df['poor_health'] == 0]['TOTEXP22'].mean()
            result['key_factors']['health_status'] = {
                'poor_health_avg': float(poor_mean),
                'good_health_avg': float(good_mean),
                'risk_multiplier': float(poor_mean / good_mean) if good_mean > 0 else 0
            }
        
        # 소득 수준 요인
        if 'is_low_income' in df.columns:
            low_income_mean = df[df['is_low_income'] == 1]['TOTEXP22'].mean()
            high_income_mean = df[df['is_low_income'] == 0]['TOTEXP22'].mean()
            result['key_factors']['income_level'] = {
                'low_income_avg': float(low_income_mean),
                'higher_income_avg': float(high_income_mean),
                'difference': float(low_income_mean - high_income_mean)
            }
        
        return result
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="리스크 분석 - 고위험 세그먼트, 리스크 스코어링, 이상치 탐지, 리스크 요인 분석",
            parameters={
                'analysis_type': {
                    'type': 'string',
                    'description': '분석 유형',
                    'enum': ['high_risk_segments', 'risk_scoring', 'outlier_detection', 'risk_factors']
                },
                'threshold': {
                    'type': 'number',
                    'description': '이상치 탐지 임계값 (default: 10000) - optional'
                }
            },
            required_params=['analysis_type']
        )

