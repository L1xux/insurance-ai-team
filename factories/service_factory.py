"""
서비스 팩토리 클래스
=========================
Author: Jin
Date: 2025.09.30
Version: 1.0

Description:
다양한 도메인별 서비스(리뷰, 보험 등)의 객체를 생성하는 팩토리 클래스입니다.
IoC Container 패턴을 활용하여 서비스 구현체를 동적으로 생성하고,
파일 타입에 따른 자동 서비스 선택 기능을 제공합니다.
"""
from enum import Enum
from typing import Dict, Type
from base.service_base import ServiceBase

from services.review.review_service import ReviewService

from services.insurance.h2023.insurance_2023_service import Insurance2023Service
from services.insurance.h2022.insurance_2022_service import Insurance2022Service

from config.logging_config import logger

class ServiceProvider(str, Enum):
    REVIEW = "review"
    INSURANCE_2022 = "insurance_2022"
    INSURANCE_2023 = "insurance_2023"


class ServiceFactory:
    """서비스 생성 및 관리 팩토리""" 
    def __init__(self):
        # 서비스 레지스트리
        self._services: Dict[str, Type[ServiceBase]] = {
            ServiceProvider.REVIEW: ReviewService,
            ServiceProvider.INSURANCE_2022: Insurance2022Service,
            ServiceProvider.INSURANCE_2023: Insurance2023Service
        }
    
    def detect_service_type(self, filepath: str) -> str:
        """
        CSV 파일을 검증하여 적절한 서비스 타입 자동 감지
        
        Args:
            filepath: 검증할 CSV 파일 경로
            
        Returns:
            감지된 서비스 타입
            
        Raises:
            ValueError: 알 수 없는 CSV 형식인 경우
        """
        for service_type, service_class in self._services.items():
            # 임시 서비스 인스턴스 생성하여 검증
            temp_service = service_class()
            
            if temp_service.validate_data(filepath):
                logger.info(f"감지된 서비스 타입: {service_type}")
                return service_type
        
        raise ValueError(f"알 수 없는 CSV 형식입니다: {filepath}")
    
    def create_service(self, service_type: str) -> ServiceBase:
        """
        서비스 인스턴스 생성
        
        Args:
            service_type: 생성할 서비스 타입
            
        Returns:
            생성된 서비스 인스턴스
            
        Raises:
            ValueError: 알 수 없는 서비스 타입인 경우
        """
        service_class = self._services.get(service_type.lower())
        
        if not service_class:
            available = ', '.join(self._services.keys())
            raise ValueError(
                f"알 수 없는 서비스 타입: {service_type}\n"
                f"사용 가능한 타입: {available}"
            )
        
        return service_class()
    
    def get_available_services(self):
        """
        등록된 서비스 목록
        
        Returns:
            사용 가능한 서비스 타입 목록
        """
        return list(self._services.keys())