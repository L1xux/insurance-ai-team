"""
데이터 분석 및 시각화 메인 실행 파일
=========================
Author: Jin
Date: 2025.09.30
Version: 1.0

Description:
CSV 파일을 선택하고 자동으로 분석 후 LLM을 통해 데이터 시각화를 생성.
"""
import sys
import os
import asyncio

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.handler.csv_handler import CsvHandler
from utils.handler.llm_model_handler import LLMModelHandler
from factories.service_factory import ServiceFactory
from services.llm.llm_service import LLMService
from config.logging_config import logger


async def main():
    """메인 함수"""
    logger.info("데이터 분석 프로젝트 시작")
    
    try:
        # 1. CSV 파일 선택
        csv_handler = CsvHandler(folder_path=current_dir)
        csv_handler.save_all_csv()
        
        selected_file = csv_handler.select_file()
        if not selected_file:
            logger.warning("파일이 선택되지 않았습니다.")
            return
        
        # 2. LLM 모델 선택
        llm_handler = LLMModelHandler()
        selection = llm_handler.select_model()
        if not selection:
            logger.warning("LLM 모델이 선택되지 않았습니다.")
            return
        
        provider, model = selection
        
        # 3. 서비스 생성 및 분석
        factory = ServiceFactory()
        service_type = factory.detect_service_type(selected_file)
        service = factory.create_service(service_type)
        
        logger.info("\n데이터 분석 중...")
        results = service.execute(selected_file)
        
        # 4. LLM 시각화
        logger.info(f"\n{provider.upper()} {model}로 시각화 생성 중...")
        llm_service = LLMService(provider=provider, model_name=model)
        
        df = service.loader.df if service.loader else None
        if df is not None:
            viz_result = await llm_service.create_visualization_with_user_input(
                results.get('analysis', {}),
                df
            )
            
            if viz_result['success']:
                print(f"\n시각화 저장 완료: {viz_result['file_path']}")
            else:
                print(f"\n시각화 생성 실패: {viz_result['error_message']}")
        
        await llm_service.close()
        logger.info("\n완료!")
        
    except KeyboardInterrupt:
        logger.info("\n프로그램이 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())