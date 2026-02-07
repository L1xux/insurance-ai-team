"""
Embedding 모델 선택 핸들러
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
사용 가능한 Embedding 모델을 조회하고 선택할 수 있는 유틸리티 클래스입니다.
모델을 선택하면 제공자는 자동으로 결정됩니다.
"""

from typing import Dict, List, Optional, Tuple


class EmbeddingModelHandler:
    """Embedding 모델 선택을 위한 핸들러"""
    
    # 모델 → 제공자 매핑
    MODEL_TO_PROVIDER: Dict[str, str] = {
        # OpenAI 모델들
        "text-embedding-3-small": "openai",
        "text-embedding-3-large": "openai",
        "text-embedding-ada-002": "openai",
        
        # Ollama 모델들
        "nomic-embed-text": "ollama",
        "mxbai-embed-large": "ollama",
        "all-minilm": "ollama",
    }
    
    # 모델 → 차원 매핑
    MODEL_TO_DIMENSION: Dict[str, int] = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        
        # Ollama
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }
    
    # 모델 설명
    MODEL_DESCRIPTIONS: Dict[str, str] = {
        # OpenAI
        "text-embedding-3-small": "OpenAI Embedding v3 Small (빠르고 효율적)",
        "text-embedding-3-large": "OpenAI Embedding v3 Large (고성능)",
        "text-embedding-ada-002": "OpenAI Ada v2 (레거시)",
        
        # Ollama
        "nomic-embed-text": "Nomic Embed Text (로컬, 빠름)",
        "mxbai-embed-large": "MxBai Embed Large (로컬, 고성능)",
        "all-minilm": "All MiniLM (로컬, 경량)",
    }
    
    def __init__(self) -> None:
        """초기화"""
        self.models: List[str] = list(self.MODEL_TO_PROVIDER.keys())
        self.selected_model: Optional[str] = None
        self.selected_dimension: Optional[int] = None
    
    def show_models(self) -> None:
        """사용 가능한 모델 목록 표시"""
        if not self.models:
            print("사용 가능한 모델이 없습니다.")
            return
        
        print("\n" + "=" * 70)
        print("사용 가능한 Embedding 모델")
        print("=" * 70)
        
        for i, model in enumerate(self.models, 1):
            provider = self.MODEL_TO_PROVIDER[model]
            dimension = self.MODEL_TO_DIMENSION[model]
            description = self.MODEL_DESCRIPTIONS.get(model, "")
            
            print(f"{i}. {model}")
            print(f"   Provider: {provider.upper()}")
            print(f"   Dimension: {dimension}")
            print(f"   {description}")
            print()
    
    def select_model(self) -> Optional[Tuple[str, str, int]]:
        """
        모델 선택
        
        Returns:
            Optional[Tuple[str, str, int]]: (provider, model, dimension) 또는 None
        """
        if not self.models:
            print("사용 가능한 모델이 없습니다.")
            return None
        
        self.show_models()
        
        try:
            choice = int(input("모델 번호를 입력하세요: ")) - 1
            
            if 0 <= choice < len(self.models):
                selected_model = self.models[choice]
                provider = self.MODEL_TO_PROVIDER[selected_model]
                dimension = self.MODEL_TO_DIMENSION[selected_model]
                
                self.selected_model = selected_model
                self.selected_dimension = dimension
                
                print(f"\n선택 완료: {provider.upper()} - {selected_model} (차원: {dimension})\n")
                return (provider, selected_model, dimension)
            else:
                print("잘못된 번호입니다.")
                return None
                
        except ValueError:
            print("숫자를 입력해주세요.")
            return None
        except KeyboardInterrupt:
            print("\n선택이 취소되었습니다.")
            return None
    
    def quick_select(self, model: str) -> Optional[Tuple[str, str, int]]:
        """
        모델명으로 직접 선택 (대화형 입력 없이)
        
        Args:
            model: 모델명 (예: 'text-embedding-3-small', 'nomic-embed-text')
        
        Returns:
            Optional[Tuple[str, str, int]]: (provider, model, dimension) 또는 None
        """
        if model not in self.MODEL_TO_PROVIDER:
            print(f"지원하지 않는 모델입니다: {model}")
            print(f"사용 가능한 모델: {', '.join(self.models)}")
            return None
        
        provider = self.MODEL_TO_PROVIDER[model]
        dimension = self.MODEL_TO_DIMENSION[model]
        self.selected_model = model
        self.selected_dimension = dimension
        
        print(f"선택 완료: {provider.upper()} - {model} (차원: {dimension})")
        return (provider, model, dimension)
    
    def get_dimension(self, model: str) -> Optional[int]:
        """
        모델의 차원 반환
        
        Args:
            model: 모델명
            
        Returns:
            차원 수 또는 None
        """
        return self.MODEL_TO_DIMENSION.get(model)

