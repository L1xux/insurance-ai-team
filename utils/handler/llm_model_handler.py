"""
LLM 모델 선택 핸들러
=========================
Author: Jin
Date: 2025.09.30
Version: 1.1 (Qwen 추가)

Description:
사용 가능한 LLM 모델을 조회하고 선택할 수 있는 유틸리티 클래스입니다.
모델을 선택하면 제공자는 자동으로 결정됩니다.
"""

from typing import Dict, List, Optional, Tuple


class LLMModelHandler:
    """LLM 모델 선택을 위한 핸들러"""
    
    # 매핑
    MODEL_TO_PROVIDER: Dict[str, str] = {
        # OpenAI 모델들
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        
        # Ollama 모델들
        "llama3.2": "ollama",
        "llama3": "ollama",
        "mistral": "ollama",
        "codellama": "ollama",
        "mixtral": "ollama",
        "phi3:mini": "ollama",
        "gemma:2b": "ollama", 
        "tinyllama": "ollama",
    }
    
    # 모델 설명
    MODEL_DESCRIPTIONS: Dict[str, str] = {
        # OpenAI
        "gpt-4o": "OpenAI GPT-4o",
        "gpt-4o-mini": "OpenAI GPT-4o Mini",
        "gpt-4-turbo": "OpenAI GPT-4 Turbo",
        "gpt-4": "OpenAI GPT-4",
        "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
        
        # Ollama
        "llama3.2": "Meta Llama 3.2",
        "llama3": "Meta Llama 3",
        "mistral": "Mistral AI",
        "codellama": "Code Llama",
        "mixtral": "Mixtral MoE",
        "phi3:mini": "Microsoft Phi-3 Mini",
        "gemma:2b": "Google Gemma 2B",
        "tinyllama": "TinyLlama",
    }
    
    def __init__(self) -> None:
        """
        초기화
        """
        self.models: List[str] = list(self.MODEL_TO_PROVIDER.keys())
        self.selected_model: Optional[str] = None
    
    def show_models(self) -> None:
        """
        사용 가능한 모델 목록 표시
        """
        if not self.models:
            print("사용 가능한 모델이 없습니다.")
            return
        
        print("\n" + "=" * 60)
        print("사용 가능한 LLM 모델")
        print("=" * 60)
        
        for i, model in enumerate(self.models, 1):
            provider = self.MODEL_TO_PROVIDER[model]
            description = self.MODEL_DESCRIPTIONS.get(model, "")
            
            print(f"{i}. {model}")
            print(f"   Provider: {provider.upper()}")
            print(f"   {description}")
            print()
    
    def select_model(self) -> Optional[Tuple[str, str]]:
        """
        모델 선택
        
        Returns:
            Optional[Tuple[str, str]]: (provider, model) 또는 None
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
                
                self.selected_model = selected_model
                
                print(f"\n선택 완료: {provider.upper()} - {selected_model}\n")
                return (provider, selected_model)
            else:
                print("잘못된 번호입니다.")
                return None
                
        except ValueError:
            print("숫자를 입력해주세요.")
            return None
        except KeyboardInterrupt:
            print("\n선택이 취소되었습니다.")
            return None
    
    def quick_select(self, model: str) -> Optional[Tuple[str, str]]:
        """
        모델명으로 직접 선택 (대화형 입력 없이)
        
        Args:
            model: 모델명 (예: 'gpt-4o', 'qwen-turbo')
        
        Returns:
            Optional[Tuple[str, str]]: (provider, model) 또는 None
        """
        if model not in self.MODEL_TO_PROVIDER:
            print(f"지원하지 않는 모델입니다: {model}")
            print(f"사용 가능한 모델: {', '.join(self.models)}")
            return None
        
        provider = self.MODEL_TO_PROVIDER[model]
        self.selected_model = model
        
        print(f"선택 완료: {provider.upper()} - {model}")
        return (provider, model)