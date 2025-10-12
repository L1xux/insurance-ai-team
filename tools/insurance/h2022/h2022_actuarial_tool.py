"""
H2022 Actuarial Analysis Tool
=========================
Author: Jin
Date: 2025.10.13
Version: 1.0

Description:
보험수리 분석 Tool (보험 준비금, 청구액 예측, 손해율 분석 등)
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


class H2022ActuarialTool(BaseTool):
    """보험수리 분석 Tool"""
    
    name: str = "h2022_actuarial_analysis"
    description: str = """보험수리 분석 도구입니다.
    
    분석 유형:
    - expected_claims: 예상 청구액 계산
    - loss_ratio: 손해율 분석
    - reserves: 준비금 계산
    - trend_analysis: 트렌드 분석
    
    입력 형식: {"analysis_type": "expected_claims", "segment": "age_group"}
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
        보험수리 분석 실행
        
        Args:
            query: JSON 문자열 또는 analysis_type
            run_manager: callback manager
            
        Returns:
            분석 결과 (JSON 문자열)
        """
        try:
            # JSON 파싱
            try:
                params = json.loads(query)
                analysis_type = params.get('analysis_type', 'expected_claims')
                segment = params.get('segment', None)
            except (json.JSONDecodeError, TypeError):
                analysis_type = query if query else 'expected_claims'
                segment = None
            
            logger.info(f"[{self.name}] 실행: {analysis_type}")
            
            if analysis_type == 'expected_claims':
                result = self._calculate_expected_claims(segment)
            elif analysis_type == 'loss_ratio':
                result = self._calculate_loss_ratio(segment)
            elif analysis_type == 'reserves':
                result = self._calculate_reserves(segment)
            elif analysis_type == 'trend_analysis':
                result = self._analyze_trends(segment)
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
    
    def _calculate_expected_claims(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """예상 청구액 계산"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        result = {
            'overall': {
                'mean_claim': float(df['TOTEXP22'].mean()),
                'median_claim': float(df['TOTEXP22'].median()),
                'total_claims': float(df['TOTEXP22'].sum()),
                'max_claim': float(df['TOTEXP22'].max())
            }
        }
        
        # 세그먼트별 분석
        if segment and segment in df.columns:
            segment_stats = df.groupby(segment)['TOTEXP22'].agg([
                ('mean', 'mean'),
                ('median', 'median'),
                ('sum', 'sum'),
                ('count', 'count')
            ]).to_dict('index')
            
            result['by_segment'] = {
                str(k): {
                    'mean_claim': float(v['mean']),
                    'median_claim': float(v['median']),
                    'total_claims': float(v['sum']),
                    'count': int(v['count'])
                }
                for k, v in segment_stats.items()
            }
        
        return result
    
    def _calculate_loss_ratio(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """손해율 분석"""
        df = self.analyzer.df
        
        # 손해율 = 지급된 청구액 / 총 보험료
        # 여기서는 간단히 총 의료비 지출 기준으로 계산
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        result = {
            'overall': {
                'total_expenditure': float(df['TOTEXP22'].sum()),
                'average_per_person': float(df['TOTEXP22'].mean()),
                'high_cost_percentage': float((df['TOTEXP22'] > 10000).sum() / len(df) * 100)
            }
        }
        
        # 세그먼트별 손해율
        if segment and segment in df.columns:
            segment_stats = df.groupby(segment).agg({
                'TOTEXP22': ['sum', 'mean', 'count']
            })
            
            result['by_segment'] = {}
            for idx in segment_stats.index:
                result['by_segment'][str(idx)] = {
                    'total_expenditure': float(segment_stats.loc[idx, ('TOTEXP22', 'sum')]),
                    'average_per_person': float(segment_stats.loc[idx, ('TOTEXP22', 'mean')]),
                    'count': int(segment_stats.loc[idx, ('TOTEXP22', 'count')])
                }
        
        return result
    
    def _calculate_reserves(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """준비금 계산 (IBNR - Incurred But Not Reported)"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        # 간단한 준비금 추정: 평균 청구액 * 청구 발생률 * 미보고 추정비율
        total_claims = df['TOTEXP22'].sum()
        claim_count = (df['TOTEXP22'] > 0).sum()
        avg_claim = df[df['TOTEXP22'] > 0]['TOTEXP22'].mean()
        
        # IBNR 추정 (실제로는 더 복잡한 actuarial 모델 사용)
        ibnr_factor = 0.15  # 15% 미보고 추정
        estimated_ibnr = total_claims * ibnr_factor
        
        result = {
            'overall': {
                'total_reported_claims': float(total_claims),
                'claim_count': int(claim_count),
                'average_claim_amount': float(avg_claim),
                'estimated_ibnr': float(estimated_ibnr),
                'total_reserve_needed': float(total_claims + estimated_ibnr)
            }
        }
        
        return result
    
    def _analyze_trends(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """트렌드 분석"""
        df = self.analyzer.df
        
        if 'TOTEXP22' not in df:
            return {'error': 'TOTEXP22 컬럼이 없습니다.'}
        
        # 지출 분포 분석
        percentiles = df['TOTEXP22'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        
        result = {
            'overall': {
                'mean': float(df['TOTEXP22'].mean()),
                'std': float(df['TOTEXP22'].std()),
                'percentiles': {f'p{int(k*100)}': float(v) for k, v in percentiles.items()},
                'zero_claims_percentage': float((df['TOTEXP22'] == 0).sum() / len(df) * 100)
            }
        }
        
        # 나이별 트렌드
        if 'AGE22X' in df.columns:
            age_trend = df.groupby('AGE22X')['TOTEXP22'].mean().to_dict()
            result['age_trend'] = {str(k): float(v) for k, v in age_trend.items() if not np.isnan(v)}
        
        return result
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="보험수리 분석 - 예상 청구액, 손해율, 준비금, 트렌드 분석",
            parameters={
                'analysis_type': {
                    'type': 'string',
                    'description': '분석 유형',
                    'enum': ['expected_claims', 'loss_ratio', 'reserves', 'trend_analysis']
                },
                'segment': {
                    'type': 'string',
                    'description': '세그먼트 기준 (예: age_group, insurance_type) - optional'
                }
            },
            required_params=['analysis_type']
        )

