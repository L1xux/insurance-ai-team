"""
LLM 시각화 서비스 통합 클래스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
데이터 분석 결과를 바탕으로 LLM을 활용한 시각화 코드 생성부터 실행까지 전체 프로세스를 관리하는 통합 서비스입니다.
사용자 입력 인터페이스, 코드 생성, 실행, 파일 저장 등의 기능을 제공하며,
VisualizationGenerator와 CodeProcessor를 조합하여 완전한 시각화 솔루션을 제공합니다.
"""
from typing import Dict, Any, Optional, List

from factories.llm_factory import LLMFactory

from .visualization_generator import VisualizationGenerator
from .code_processor import CodeProcessor
from config.logging_config import logger


class LLMService:
    """데이터 분석 결과를 바탕으로 시각화 코드를 생성하고 실행하는 통합 서비스"""
    
    def __init__(
        self, 
        provider: Optional[str], 
        model_name: Optional[str],
        output_dir: str = "data"
    ):
        """
        Args:
            provider: LLM provider (예: 'openai', 'anthropic', 'google')
            model_name: model (예: 'gpt-4o', 'claude-sonnet-4.5')
            output_dir: 시각화 결과 저장 디렉토리
        """
        # LLM Factory를 통한 LLM 객체 생성
        factory = LLMFactory()
        self.llm = factory.create_llm(provider, model_name)    
        
        # LLM 객체를 주입하여 Generator와 Processor 초기화
        self.generator = VisualizationGenerator(llm=self.llm)
        self.processor = CodeProcessor(output_dir)
        
        logger.info(f"LLMService 초기화 완료: {self.llm}, 출력={output_dir}")
        
    
    def get_user_instructions(self, available_fields: Optional[List[str]] = None) -> str:
        """
        사용자로부터 시각화 지시문을 입력받는 인터페이스
        
        Args:
            available_fields: 사용 가능한 데이터 필드 목록
            
        Returns:
            사용자가 입력한 지시문
        """
        print("=" * 60)
        print("데이터 시각화 지시문 입력")
        print("=" * 60)
        
        if available_fields:
            print(f"사용 가능한 데이터 필드: {', '.join(available_fields)}")
            print()
        
        print("어떤 시각화를 원하시나요? 다음과 같이 입력해주세요:")
        print("예시:")
        print("- '카테고리별 평점 분포를 막대 그래프로 보여주세요'")
        print("- '평점과 가격의 상관관계를 산점도로 표현해주세요'")
        print("- '시간에 따른 판매량 변화를 선 그래프로 그려주세요'")
        print("- '이상치를 포함한 데이터 분포를 박스플롯으로 시각화해주세요'")
        print()
        
        # 빈 입력시 기본 지시문 제공
        user_input = input("지시문을 입력하세요 (Enter만 누르면 자동 분석): ").strip()
        
        if not user_input:
            user_input = "데이터의 특성을 분석하여 가장 유의미한 시각화를 자동으로 생성해주세요"
            print(f"자동 분석 모드: {user_input}")
        else:
            print(f"입력된 지시문: {user_input}")
        
        print("=" * 60)
        return user_input
    
    async def create_visualization_with_user_input(
        self,
        analysis_results: Dict[str, Any],
        df: Any,  # DataFrame
        save_plot: bool = True,
        filename: Optional[str] = None,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        사용자 입력을 받아서 시각화를 생성하는 대화형 메소드
        
        Args:
            analysis_results: 데이터 분석 결과
            df: 시각화할 DataFrame
            save_plot: 플롯 저장 여부
            filename: 저장할 파일명
            interactive: 대화형 모드 사용 여부
            
        Returns:
            Dict: {
                'success': bool,
                'generated_code': str,
                'file_path': str,
                'error_message': str,
                'user_context': str
            }
        """
        result: Dict[str, Any] = {
            'success': False,
            'generated_code': None,
            'file_path': None,
            'error_message': None,
            'user_context': None
        }
        
        try:
            logger.info("대화형 시각화 생성 프로세스 시작")
            
            # DataFrame 유효성 검사
            df_valid, df_error = self.processor.validate_dataframe(df)
            if not df_valid:
                result['error_message'] = df_error
                return result
            
            # 사용 가능한 필드 추출
            available_fields = []
            if 'columns_info' in analysis_results:
                available_fields = [col['name'] for col in analysis_results['columns_info']]
            elif hasattr(df, 'columns'):
                available_fields = df.columns.tolist()
            
            # 사용자 입력 받기
            if interactive:
                user_context = self.get_user_instructions(available_fields)
            else:
                user_context = "데이터의 특성을 분석하여 가장 유의미한 시각화를 자동으로 생성해주세요"
            
            result['user_context'] = user_context
            
            # 시각화 코드 생성
            logger.info("사용자 지시문을 바탕으로 시각화 코드 생성")
            generated_code = await self.generator.generate_visualization_code(
                analysis_results, user_context, None
            )
            result['generated_code'] = generated_code
            
            # 코드 실행
            logger.info("생성된 코드 실행")
            execution_success, file_path, execution_error = self.processor.execute_visualization_code(
                generated_code, df, save_plot, filename
            )
            
            if execution_success:
                result['success'] = True
                result['file_path'] = file_path
                logger.info(f"대화형 시각화 생성 완료: {file_path}")
                
                if interactive:
                    print(f"시각화가 성공적으로 생성되었습니다!")
                    print(f"저장 위치: {file_path}")
            else:
                result['error_message'] = execution_error
                logger.error(f"코드 실행 실패: {execution_error}")
                
                if interactive:
                    print(f"시각화 생성 중 오류가 발생했습니다: {execution_error}")
            
        except Exception as e:
            error_msg = f"대화형 시각화 생성 프로세스 실패: {str(e)}"
            result['error_message'] = error_msg
            logger.error(error_msg)
            
            if interactive:
                print(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        
        return result
    
    async def create_visualization_from_analysis(
        self,
        analysis_results: Dict[str, Any],
        df: Any,  # DataFrame
        user_context: Optional[str] = None,
        specific_fields: Optional[List[str]] = None,
        save_plot: bool = True,
        filename: Optional[str] = None,
        request_user_input: bool = False
    ) -> Dict[str, Any]:
        """
        데이터 분석 결과를 바탕으로 시각화 생성 (전체 프로세스)
        
        Args:
            analysis_results: 데이터 분석 결과
            df: 시각화할 DataFrame
            user_context: 사용자 컨텍스트 (예: "카테고리별 평점 분석")
            specific_fields: 특정 필드들 지정
            save_plot: 플롯 저장 여부
            filename: 저장할 파일명
            request_user_input: True이면 사용자로부터 지시문 입력받음
            
        Returns:
            Dict: {
                'success': bool,
                'generated_code': str,
                'file_path': str,
                'error_message': str,
                'user_context': str
            }
        """
        result: Dict[str, Any] = {
            'success': False,
            'generated_code': None,
            'file_path': None,
            'error_message': None,
            'user_context': user_context
        }
        
        try:
            logger.info("시각화 생성 프로세스 시작")
            
            # 1. DataFrame 유효성 검사
            df_valid, df_error = self.processor.validate_dataframe(df)
            if not df_valid:
                result['error_message'] = df_error
                return result
            
            # 2. 사용자 입력 처리
            if request_user_input and not user_context:
                # 사용 가능한 필드 추출
                available_fields = []
                if 'columns_info' in analysis_results:
                    available_fields = [col['name'] for col in analysis_results['columns_info']]
                elif hasattr(df, 'columns'):
                    available_fields = df.columns.tolist()
                
                user_context = self.get_user_instructions(available_fields)
            
            result['user_context'] = user_context
            
            # 3. 시각화 코드 생성
            logger.info("LLM을 통한 시각화 코드 생성")
            generated_code = await self.generator.generate_visualization_code(
                analysis_results, user_context, specific_fields
            )
            result['generated_code'] = generated_code
            
            # 3. 코드 실행
            logger.info("생성된 코드 실행")
            execution_success, file_path, execution_error = self.processor.execute_visualization_code(
                generated_code, df, save_plot, filename
            )
            
            if execution_success:
                result['success'] = True
                result['file_path'] = file_path
                logger.info(f"시각화 생성 완료: {file_path}")
            else:
                result['error_message'] = execution_error
                logger.error(f"코드 실행 실패: {execution_error}")
            
        except Exception as e:
            error_msg = f"시각화 생성 프로세스 실패: {str(e)}"
            result['error_message'] = error_msg
            logger.error(error_msg)
        
        return result
    
    async def generate_code_only(
        self,
        analysis_results: Dict[str, Any],
        user_context: Optional[str] = None,
        specific_fields: Optional[List[str]] = None,
        request_user_input: bool = False
    ) -> Dict[str, Any]:
        """
        코드 생성만 수행 (실행하지 않음)
        
        Args:
            analysis_results: 데이터 분석 결과
            user_context: 사용자 컨텍스트
            specific_fields: 특정 필드들 지정
            request_user_input: True이면 사용자로부터 지시문 입력받음
        
        Returns:
            Dict: {
                'success': bool,
                'generated_code': str,
                'error_message': str,
                'user_context': str
            }
        """
        result: Dict[str, Any] = {
            'success': False,
            'generated_code': None,
            'error_message': None,
            'user_context': None
        }
        
        try:
            logger.info("시각화 코드 생성 (실행 안함)")
            
            # 사용자 입력 처리
            final_user_context = user_context
            if request_user_input and not user_context:
                # 사용 가능한 필드 추출
                available_fields = []
                if 'columns_info' in analysis_results:
                    available_fields = [col['name'] for col in analysis_results['columns_info']]
                
                final_user_context = self.get_user_instructions(available_fields)
            
            result['user_context'] = final_user_context
            
            generated_code = await self.generator.generate_visualization_code(
                analysis_results, final_user_context, specific_fields
            )
            
            result['success'] = True
            result['generated_code'] = generated_code
            
        except Exception as e:
            result['error_message'] = f"코드 생성 실패: {str(e)}"
            logger.error(result['error_message'])
        
        return result
    
    def execute_custom_code(
        self,
        code: str,
        df: Any,  # DataFrame
        save_plot: bool = True,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        사용자 제공 코드 실행
        
        Returns:
            Dict: {
                'success': bool,
                'file_path': str,
                'error_message': str
            }
        """
        result: Dict[str, Any] = {
            'success': False,
            'file_path': None,
            'error_message': None
        }
        
        try:
            logger.info("사용자 제공 코드 실행")
            
            # DataFrame 유효성 검사
            df_valid, df_error = self.processor.validate_dataframe(df)
            if not df_valid:
                result['error_message'] = df_error
                return result
            
            # 코드 실행
            execution_success, file_path, execution_error = self.processor.execute_visualization_code(
                code, df, save_plot, filename
            )
            
            if execution_success:
                result['success'] = True
                result['file_path'] = file_path
            else:
                result['error_message'] = execution_error
            
        except Exception as e:
            result['error_message'] = f"사용자 코드 실행 실패: {str(e)}"
            logger.error(result['error_message'])
        
        return result
    
    def get_available_examples(self) -> Dict[str, str]:
        """
        사용 가능한 시각화 예시들 반환
        
        Returns:
            시각화 예시 딕셔너리
        """
        examples = {
            "카테고리별 분포": "카테고리별 데이터 분포를 막대 그래프로 시각화",
            "평점 분석": "평점 분포와 카테고리별 평균 평점 비교",
            "상관관계 분석": "수치형 데이터 간의 상관관계 히트맵",
            "트렌드 분석": "시간에 따른 변화 추이 분석",
            "이상치 탐지": "박스플롯을 통한 이상치 시각화",
            "분포 비교": "여러 그룹 간의 분포 비교",
            "다차원 분석": "여러 변수를 동시에 고려한 복합 시각화"
        }
        return examples
    
    def get_usage_examples(self) -> str:
        """
        사용자 입력 기능을 포함한 사용법 예시 반환
        
        Returns:
            사용법 예시 문자열
        """
        examples = """
            LLMService의 메소드를 사용하여 시각화를 생성할 수 있습니다.
        """
        return examples.strip()
    
    def get_output_files(self) -> List[str]:
        """
        생성된 시각화 파일 목록 반환
        
        Returns:
            생성된 파일 경로 목록
        """
        return self.processor.get_output_files()
    
    async def close(self) -> None:
        """
        리소스 정리
        """
        await self.generator.close()
        logger.info("LLMService 리소스 정리 완료")
