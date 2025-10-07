"""
프로젝트 설정 관리자
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
프로젝트 전체의 설정값들을 관리하는 모듈입니다.
경로 설정, 데이터 처리 옵션, 로그 레벨, 시각화 스타일 등
프로젝트 운영에 필요한 모든 설정을 중앙에서 관리합니다.
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class _Settings:
    """간단한 설정 클래스"""
    
    # 기본 경로
    project_root: Path = Path(__file__).parent.parent
    data_path: Path = project_root / "data"
    logs_path: Path = project_root
    
    # 데이터 처리 설정
    max_chunk_size: int = 10000
    
    # 로그 설정
    log_level: str = "INFO"
    
    # 시각화 설정
    figure_size: tuple = (12, 8)
    plot_style: str = "seaborn-v0_8"
    
    def __post_init__(self):
        """
        필요한 디렉토리 생성
        
        로그 디렉토리가 존재하지 않는 경우 생성합니다.
        """
        self.logs_path.mkdir(exist_ok=True)


# 전역 설정 인스턴스
settings = _Settings()