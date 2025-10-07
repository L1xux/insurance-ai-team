"""
로깅 설정 관리자
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
프로젝트 전체의 로깅 설정을 관리하는 모듈입니다.
콘솔과 파일 출력을 지원하며, 설정 기반의 로그 레벨 관리,
포맷팅, 핸들러 설정 등의 기능을 제공합니다.
"""
import logging
import sys

from .settings import settings


def setup_logging(name: str = "DataAnalysis") -> logging.Logger:
    """
    간단한 로깅 설정
    
    Args:
        name: 로거 이름
        
    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))
    
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if settings.logs_path.exists():
        file_handler = logging.FileHandler(settings.logs_path / f"{name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 기본 로거
logger = setup_logging()