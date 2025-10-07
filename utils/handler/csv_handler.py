"""
CSV 파일 핸들러
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
CSV 파일을 검색하고 선택할 수 있는 유틸리티 클래스입니다.
지정된 폴더에서 CSV 파일들을 자동으로 찾아 목록을 제공하며,
사용자가 원하는 파일을 선택할 수 있는 인터페이스를 제공합니다.
"""

import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Optional

class CsvHandler:
    def __init__(self, folder_path: str = "") -> None:
        self.folder_path: Path = Path(folder_path)
        self.csv_files: List[str] = []
        
    def save_all_csv(self) -> List[str]:
        """
        지정된 폴더에서 모든 CSV 파일을 찾아 목록에 저장
        
        Returns:
            찾은 CSV 파일 경로 목록
        """
        patterns: str = str(self.folder_path / "data" / "*.csv")        
        self.csv_files: List[str] = glob.glob(patterns, recursive=True)
    
        return self.csv_files
    
    def show_csv_list(self) -> None:
        """
        저장된 CSV 파일 목록을 화면에 표시
        """
        if not self.csv_files:
            print("CSV 파일이 없습니다.")
            return
        
        for i, file_path in enumerate(self.csv_files, 1):
            name: str = os.path.basename(file_path)
            size: int = os.path.getsize(file_path)
            
            print(f"{i}. {file_path}")
            print(f"name: {name}")
            print(f"path: {file_path}")
            print(f"size: {size:,} bytes")
            print()
    
    def quick_check(self, file_path: str) -> bool:
        """
        CSV 파일 빠른 확인
        
        Args:
            file_path: 확인할 CSV 파일 경로
            
        Returns:
            파일 읽기 성공 여부
        """
        try: 
            df: pd.DataFrame = pd.read_csv(file_path, nrows=5)
            
            print(f"파일: {os.path.basename(file_path)}")
            print(f"   행수: {len(df)} (샘플)")
            print(f"   열수: {len(df.columns)}")
            print(f"   컬럼: {list(df.columns)}")
            print(f"   미리보기:")
            print(df.to_string(index=False))
            print("-" * 50)

            return True
            
        except Exception as e:
            print(f"오류: {e}")
            return False
        
    def select_file(self) -> Optional[str]:
        """
        사용자가 선택한 CSV 파일 반환
        
        Returns:
            선택된 파일 경로 또는 None
        """
        if not self.csv_files:
            print("save csv files at first")
            return None
        
        self.show_csv_list()
        
        try:
            choice: int = int(input("확인할 파일 번호를 입력하세요: ")) - 1
            
            if 0 <= choice < len(self.csv_files):
                selected_file: str = self.csv_files[choice]
                self.quick_check(selected_file)
                return selected_file
            else:
                print("잘못된 번호입니다.")
                return None
                
        except ValueError:
            print("숫자를 입력해주세요.")
            return None