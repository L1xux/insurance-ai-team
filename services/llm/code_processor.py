"""
LLM 생성 코드 실행 프로세서
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
LLM이 생성한 시각화 코드를 안전하게 실행하고 결과를 저장하는 클래스입니다.
subprocess를 활용하여 코드를 격리된 환경에서 실행하고, 보안 검증, 타임아웃 처리,
오류 핸들링 등의 안전 장치를 제공합니다. 생성된 시각화를 이미지 파일로 저장합니다.
"""
import os
import sys
import tempfile
import subprocess
from typing import Optional, Tuple
import pandas as pd
from pathlib import Path

from config.logging_config import logger


class CodeProcessor:
    """LLM이 생성한 시각화 코드를 안전하게 실행하는 클래스"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 시각화 결과를 저장할 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 안전한 실행을 위한 허용된 모듈들
        self.allowed_modules = {
            'matplotlib', 'matplotlib.pyplot', 'plt',
            'seaborn', 'sns', 
            'pandas', 'pd',
            'numpy', 'np',
            'warnings'
        }
        
        logger.info(f"CodeProcessor 초기화: 출력 디렉토리 = {self.output_dir}")
    
    def execute_visualization_code(
        self, 
        code: str, 
        df: pd.DataFrame,
        save_plot: bool = True,
        filename: Optional[str] = None,
        timeout: int = 30
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        시각화 코드를 subprocess로 안전하게 실행
        
        Args:
            code: 실행할 Python 코드
            df: 시각화에 사용할 DataFrame
            save_plot: 플롯을 파일로 저장할지 여부
            filename: 저장할 파일명 (없으면 자동 생성)
            timeout: 실행 타임아웃 (초)
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
                (성공 여부, 저장된 파일 경로, 에러 메시지)
        """
        try:
            logger.info("subprocess를 통한 시각화 코드 실행 시작")
            
            # 코드 안전성 검사
            if not self._is_code_safe(code):
                error_msg = "안전하지 않은 코드가 감지되었습니다."
                logger.error(error_msg)
                return False, None, error_msg
            
            # 파일명 결정
            if not filename:
                filename = f"visualization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            output_path = self.output_dir / "visualizations" / filename
            
            # 임시 파일들 생성
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # DataFrame을 pickle로 저장
                df_path = temp_path / "dataframe.pkl"
                df.to_pickle(df_path)
                
                # 실행할 Python 스크립트 생성
                script_path = temp_path / "visualization_script.py"
                script_content = self._create_subprocess_script(code, df_path, output_path, save_plot)
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                
                # subprocess로 실행
                success, error_message = self._run_subprocess(script_path, timeout)
                
                if success:
                    if save_plot and output_path.exists():
                        logger.info(f"시각화 저장 완료: {output_path}")
                        return True, str(output_path), None
                    else:
                        logger.info("시각화 실행 완료")
                        return True, None, None
                else:
                    return False, None, error_message
                    
        except Exception as e:
            error_message = f"subprocess 실행 중 오류 발생: {str(e)}"
            logger.error(error_message)
            return False, None, error_message
    
    def _is_code_safe(self, code: str) -> bool:
        """
        코드 안전성 검사
        
        Args:
            code: 검사할 코드 문자열
            
        Returns:
            코드가 안전한지 여부
        """
        # 위험한 키워드들 체크
        dangerous_keywords = [
            'import os', 'import sys', 'import subprocess',
            'exec(', 'eval(', '__import__', 
            'open(', 'file(', 'input(', 'raw_input(',
            'delete', 'remove', 'rmdir',
            'system', 'popen', 'spawn'
        ]
        
        code_lower = code.lower()
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                logger.warning(f"위험한 키워드 감지: {keyword}")
                return False
        
        # 허용된 import만 체크
        import_lines = [line.strip() for line in code.split('\\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        for line in import_lines:
            if not self._is_import_allowed(line):
                logger.warning(f"허용되지 않은 import: {line}")
                return False
        
        return True
    
    def _is_import_allowed(self, import_line: str) -> bool:
        """
        import 문이 허용되는지 확인
        
        Args:
            import_line: 검사할 import 문
            
        Returns:
            import가 허용되는지 여부
        """
        # 기본 허용된 모듈들
        for allowed in self.allowed_modules:
            if allowed in import_line:
                return True
        
        # 추가 허용 패턴들
        allowed_patterns = [
            'from matplotlib',
            'from seaborn', 
            'from pandas',
            'from numpy',
            'import warnings'
        ]
        
        for pattern in allowed_patterns:
            if pattern in import_line:
                return True
                
        return False
    
    def _create_subprocess_script(self, code: str, df_path: Path, output_path: Path, save_plot: bool) -> str:
        """
        subprocess에서 실행할 스크립트 생성
        
        Args:
            code: 실행할 사용자 코드
            df_path: DataFrame pickle 파일 경로
            output_path: 출력 파일 경로
            save_plot: 플롯 저장 여부
            
        Returns:
            생성된 스크립트 내용
        """
        script_template = f'''
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    # DataFrame 로드
    df = pd.read_pickle(r"{df_path}")
    
    # 사용자 코드 실행
{self._indent_code(code, 4)}
    
    # create_visualization 함수 호출
    if 'create_visualization' in locals():
        fig = create_visualization(df)
        
        if fig and {save_plot}:
            # 파일 저장
            fig.savefig(r"{output_path}", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("SUCCESS: Visualization saved")
        else:
            print("SUCCESS: Visualization completed")
    else:
        print("ERROR: create_visualization function not found")
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        return script_template
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """
        코드를 지정된 공백으로 들여쓰기
        
        Args:
            code: 들여쓸 코드
            spaces: 들여쓰기 공백 수
            
        Returns:
            들여쓰기된 코드
        """
        indent = ' ' * spaces
        lines = code.split('\n')
        indented_lines = [indent + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    def _run_subprocess(self, script_path: Path, timeout: int) -> Tuple[bool, Optional[str]]:
        """
        subprocess로 스크립트 실행
        
        Args:
            script_path: 실행할 스크립트 경로
            timeout: 실행 타임아웃 (초)
            
        Returns:
            (성공 여부, 에러 메시지)
        """
        try:
            # Python 실행 명령어
            python_cmd = [sys.executable, str(script_path)]
            
            # subprocess 실행
            result = subprocess.run(
                python_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            # 결과 확인
            if result.returncode == 0:
                if "SUCCESS" in result.stdout:
                    logger.info("subprocess 실행 성공")
                    return True, None
                else:
                    error_msg = f"스크립트 실행 중 예상치 못한 출력: {result.stdout}"
                    logger.error(error_msg)
                    return False, error_msg
            else:
                error_msg = f"subprocess 실행 실패 (코드: {result.returncode})"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nSTDOUT: {result.stdout}"
                logger.error(error_msg)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = f"subprocess 실행 타임아웃 ({timeout}초)"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"subprocess 실행 중 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        DataFrame 유효성 검사
        
        Args:
            df: 검사할 DataFrame
            
        Returns:
            (유효성 여부, 에러 메시지)
        """
        if df is None:
            return False, "DataFrame이 None입니다."
        
        if df.empty:
            return False, "DataFrame이 비어있습니다."
        
        if len(df.columns) == 0:
            return False, "DataFrame에 컬럼이 없습니다."
        
        logger.info(f"DataFrame 검증 완료: {df.shape}")
        return True, None
    
    def create_code_template(self, columns: list) -> str:
        """
        기본 코드 템플릿 생성
        
        Args:
            columns: 사용 가능한 컬럼 목록
            
        Returns:
            생성된 코드 템플릿
        """
        template = f'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_visualization(df):
    """
    사용 가능한 컬럼: {', '.join(columns)}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 여기에 시각화 코드 작성
    # 예시: df['column_name'].plot(kind='bar', ax=ax)
    
    plt.title('데이터 시각화')
    plt.tight_layout()
    return fig
'''
        return template.strip()
    
    def get_output_files(self) -> list:
        """
        생성된 시각화 파일 목록 반환
        
        Returns:
            생성된 파일 경로 목록
        """
        if not self.output_dir.exists():
            return []
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg']
        files = []
        
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                files.append(str(file_path))
        
        return sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
