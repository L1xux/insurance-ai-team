"""
통합 보험 데이터 로더
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
data/ 폴더의 모든 보험 CSV 파일을 병렬로 로드하는 통합 로더입니다.
concurrent.futures를 사용하여 여러 파일을 동시에 처리합니다.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from utils.data.insurance_2022_data_loader import Insurance2022DataLoader
from utils.data.insurance_2023_data_loader import Insurance2023DataLoader
from models.insurance_model import InsuranceDataBatch
from config.logging_config import logger


class AllInsuranceCSVLoader:
    """모든 보험 CSV 파일을 병렬로 로드하는 통합 로더"""
    
    # 파일명 → 로더 매핑
    FILE_TO_LOADER = {
        'insurance_US_2022.csv': Insurance2022DataLoader,
        'insurance_US_2023.csv': Insurance2023DataLoader,
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        통합 로더 초기화
        
        Args:
            data_dir: CSV 파일이 있는 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.loaders: Dict[str, object] = {}
        self.batches: Dict[str, InsuranceDataBatch] = {}
        
        logger.info(f"[AllInsuranceCSVLoader] 초기화 (data_dir={self.data_dir})")
    
    def _load_single_file(self, filepath: Path) -> Tuple[str, Optional[InsuranceDataBatch]]:
        """
        단일 파일을 로드 (병렬 처리용 헬퍼 함수)
        
        Args:
            filepath: CSV 파일 경로
            
        Returns:
            (파일명, InsuranceDataBatch 또는 None)
        """
        filename = filepath.name
        
        try:
            # 파일명에 맞는 로더 선택
            loader_class = None
            for pattern, loader_cls in self.FILE_TO_LOADER.items():
                if pattern in filename:
                    loader_class = loader_cls
                    break
            
            if not loader_class:
                logger.warning(f"[AllInsuranceCSVLoader] 지원하지 않는 파일: {filename}")
                return (filename, None)
            
            # 로더 생성 및 데이터 로드
            loader = loader_class()
            logger.info(f"[AllInsuranceCSVLoader] 로드 시작: {filename}")
            
            batch = loader.process(str(filepath))
            
            if batch:
                logger.info(
                    f"[AllInsuranceCSVLoader] 로드 완료: {filename} "
                    f"({batch.size:,}개 레코드)"
                )
                return (filename, batch)
            else:
                logger.warning(f"[AllInsuranceCSVLoader] 빈 배치: {filename}")
                return (filename, None)
                
        except Exception as e:
            logger.error(f"[AllInsuranceCSVLoader] 로드 실패: {filename} - {str(e)}")
            return (filename, None)
    
    def load_all_parallel(
        self,
        max_workers: Optional[int] = None
    ) -> Dict[str, InsuranceDataBatch]:
        """
        모든 CSV 파일을 병렬로 로드
        
        Args:
            max_workers: 최대 워커 수 (None이면 CPU 코어 수만큼)
            
        Returns:
            파일명 → InsuranceDataBatch 딕셔너리
        """
        try:
            # CSV 파일 목록 수집
            csv_files = list(self.data_dir.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"[AllInsuranceCSVLoader] CSV 파일 없음: {self.data_dir}")
                return {}
            
            logger.info(
                f"[AllInsuranceCSVLoader] 병렬 로드 시작: {len(csv_files)}개 파일 "
                f"(max_workers={max_workers or 'auto'})"
            )
            
            # 병렬 처리
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 파일에 대해 작업 제출
                future_to_file = {
                    executor.submit(self._load_single_file, filepath): filepath
                    for filepath in csv_files
                }
                
                # 완료된 작업부터 처리
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        filename, batch = future.result()
                        if batch:
                            results[filename] = batch
                            self.batches[filename] = batch
                    except Exception as e:
                        logger.error(
                            f"[AllInsuranceCSVLoader] 예외 발생: {filepath.name} - {str(e)}"
                        )
            
            # 결과 요약
            total_records = sum(batch.size for batch in results.values())
            logger.info(
                f"[AllInsuranceCSVLoader] 병렬 로드 완료: "
                f"{len(results)}/{len(csv_files)}개 파일 성공, "
                f"총 {total_records:,}개 레코드"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"[AllInsuranceCSVLoader] 병렬 로드 실패: {str(e)}")
            return {}
    
    def load_all_sequential(self) -> Dict[str, InsuranceDataBatch]:
        """
        모든 CSV 파일을 순차적으로 로드 (폴백용)
        
        Returns:
            파일명 → InsuranceDataBatch 딕셔너리
        """
        try:
            csv_files = list(self.data_dir.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"[AllInsuranceCSVLoader] CSV 파일 없음: {self.data_dir}")
                return {}
            
            logger.info(f"[AllInsuranceCSVLoader] 순차 로드 시작: {len(csv_files)}개 파일")
            
            results = {}
            for filepath in csv_files:
                filename, batch = self._load_single_file(filepath)
                if batch:
                    results[filename] = batch
                    self.batches[filename] = batch
            
            total_records = sum(batch.size for batch in results.values())
            logger.info(
                f"[AllInsuranceCSVLoader] 순차 로드 완료: "
                f"{len(results)}/{len(csv_files)}개 파일, "
                f"총 {total_records:,}개 레코드"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"[AllInsuranceCSVLoader] 순차 로드 실패: {str(e)}")
            return {}
    
    def get_batch(self, filename: str) -> Optional[InsuranceDataBatch]:
        """
        특정 파일의 배치 가져오기
        
        Args:
            filename: 파일명
            
        Returns:
            InsuranceDataBatch 또는 None
        """
        return self.batches.get(filename)
    
    def get_all_batches(self) -> Dict[str, InsuranceDataBatch]:
        """
        모든 배치 가져오기
        
        Returns:
            파일명 → InsuranceDataBatch 딕셔너리
        """
        return self.batches.copy()
    
    def get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """
        모든 배치의 DataFrame을 하나로 합치기
        
        Returns:
            결합된 DataFrame 또는 None
        """
        try:
            if not self.batches:
                logger.warning("[AllInsuranceCSVLoader] 로드된 배치가 없습니다.")
                return None
            
            # 각 배치에서 DataFrame 추출
            dfs = []
            for filename, batch in self.batches.items():
                df = batch.to_dataframe()
                df['source_file'] = filename  # 출처 파일 표시
                dfs.append(df)
            
            # DataFrame 결합
            combined_df = pd.concat(dfs, ignore_index=True)
            
            logger.info(
                f"[AllInsuranceCSVLoader] DataFrame 결합 완료: "
                f"{len(dfs)}개 파일, 총 {len(combined_df):,}개 레코드"
            )
            
            return combined_df
            
        except Exception as e:
            logger.error(f"[AllInsuranceCSVLoader] DataFrame 결합 실패: {str(e)}")
            return None
    
    def get_summary(self) -> Dict[str, any]:
        """
        로드된 데이터 요약 정보
        
        Returns:
            요약 정보 딕셔너리
        """
        summary = {
            'total_files': len(self.batches),
            'files': {},
            'total_records': 0
        }
        
        for filename, batch in self.batches.items():
            # 컬럼 수 계산 (첫 번째 insurance의 필드 수)
            columns_count = 0
            if batch.size > 0 and batch.insurances:
                columns_count = len(batch.insurances[0].get_fields())
            
            summary['files'][filename] = {
                'records': batch.size,
                'columns': columns_count
            }
            summary['total_records'] += batch.size
        
        return summary

