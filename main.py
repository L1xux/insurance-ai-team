"""
Agent CLI
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
Manager Agent와 CLI로 상호작용하는 인터페이스입니다.
"""
import asyncio
import sys
from pathlib import Path

from factories.agent_factory import get_agent_factory
from agents.orchestrator.manager_agent import InsuranceManagerAgent
from agents.team.customer_insight.customer_insight_agent import CustomerInsightAgent
from tools.insurance.h2022 import H2022DemographicTool, H2022ActuarialTool, H2022RiskTool
from tools.rag import MultiQueryRAGTool, HyDERAGTool, ContextualCompressionRAGTool, SelfQueryRAGTool
from tools.code_generator import VisualizationTool
from rag.common.retriever import Retriever
from rag.insurance.h2022.h2022_embedding import H2022Embedding
from rag.insurance.h2022.h2022_vector_store import H2022VectorStore
from rag.common.openai_embedder import OpenAIEmbedder
from base.agent.llm_base import LLMBase
from config.logging_config import logger
from utils.data.insurance_2022_data_analyzer import Insurance2022DataAnalyzer
from utils.data.insurance_2022_data_loader import Insurance2022DataLoader
from utils.data.all_insurance_csv_loader import AllInsuranceCSVLoader
from agents.llm_providers.llm_openai import LLMOpenAI
from agents.llm_providers.llm_ollama import LLMOllama
from utils.handler.llm_model_handler import LLMModelHandler
from utils.handler.embedding_model_handler import EmbeddingModelHandler
from rag.common.ollama_embedder import OllamaEmbedder


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print("Insurance Data Analysis Agent System")
    print("=" * 70 + "\n")


def print_section(title: str):
    """섹션 헤더 출력"""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")


def select_embedding_model() -> tuple:
    """Embedding 모델 선택"""
    print("\nRAG를 위한 Embedding 모델 선택")
    print("   (Enter를 누르면 기본값 'text-embedding-3-small' 사용)\n")
    
    handler = EmbeddingModelHandler()
    
    # 기본값 옵션 제공
    choice = input("모델 목록을 보시겠습니까? (y/n, 또는 Enter로 기본값 사용): ").strip().lower()
    
    if not choice or choice == 'n':
        print("   text-embedding-3-small 선택됨 (기본값)")
        return ("openai", "text-embedding-3-small", 1536)
    
    result = handler.select_model()
    if result:
        return result
    else:
        print("   text-embedding-3-small 선택됨 (기본값)")
        return ("openai", "text-embedding-3-small", 1536)


def create_llm_from_selection(provider: str, model: str, temperature: float = 0.7):
    """
    Provider와 모델명으로 LLM 구현체 생성
    
    Args:
        provider: 제공자 (openai, ollama)
        model: 모델 이름
        temperature: 생성 온도
        
    Returns:
        LLMBase 구현체 (LLMOpenAI 또는 LLMOllama)
    """
    if provider == "openai":
        return LLMOpenAI(model_name=model, temperature=temperature)
    elif provider == "ollama":
        return LLMOllama(model_name=model, temperature=temperature)
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")


def select_llm_with_handler(purpose: str) -> tuple:
    """
    LLMModelHandler를 사용하여 LLM 선택
    
    Args:
        purpose: 사용 목적 (예: "Manager Agent")
        
    Returns:
        (provider, model) 튜플
    """
    print(f"\n{purpose}을 위한 LLM 선택")
    print("   (Enter를 누르면 기본값 'gpt-4o-mini' 사용)\n")
    
    handler = LLMModelHandler()
    
    # 기본값 옵션 제공
    choice = input("모델 목록을 보시겠습니까? (y/n, 또는 Enter로 기본값 사용): ").strip().lower()
    
    if not choice or choice == 'n':
        print("   gpt-4o-mini 선택됨 (기본값)")
        return ("openai", "gpt-4o-mini")
    
    result = handler.select_model()
    if result:
        return result
    else:
        print("   gpt-4o-mini 선택됨 (기본값)")
        return ("openai", "gpt-4o-mini")


def select_team_llm_with_handler() -> dict:
    """
    LLMModelHandler를 사용하여 Team별 LLM 선택
    
    Returns:
        팀별 (provider, model) 튜플 딕셔너리
    """
    print("\nWorker Agent Teams:")
    print("   1. Customer Insight Team (인구통계, 구매행동 분석)")
    print("\n   모든 Teams에 동일한 LLM 사용")
    print("   (Enter를 누르면 기본값 'gpt-4o-mini' 사용)\n")
    
    handler = LLMModelHandler()
    
    choice = input("모델 목록을 보시겠습니까? (y/n, 또는 Enter로 기본값 사용): ").strip().lower()
    
    if not choice or choice == 'n':
        print("   gpt-4o-mini 선택됨 (기본값)")
        result = ("openai", "gpt-4o-mini")
    else:
        result = handler.select_model()
        if not result:
            print("   gpt-4o-mini 선택됨 (기본값)")
            result = ("openai", "gpt-4o-mini")
    
    return {
        'customer_insight': result
    }


async def setup_agents():
    """Agent 시스템 설정"""
    print_section("Agent 시스템 설정")
    
    # LLM 선택 (LLMModelHandler 사용)
    manager_provider, manager_model = select_llm_with_handler("Manager Agent")
    team_llms = select_team_llm_with_handler()
    
    # Factory
    factory = get_agent_factory()
    
    # LLM 등록
    manager_llm = create_llm_from_selection(manager_provider, manager_model, temperature=0.7)
    customer_provider, customer_model = team_llms['customer_insight']
    customer_llm = create_llm_from_selection(customer_provider, customer_model, temperature=0.7)
    
    factory.register_llm("manager_llm", manager_llm)
    factory.register_llm("customer_insight_llm", customer_llm)
    
    print(f"\nLLM 등록 완료")
    print(f"   Manager: {manager_provider.upper()} - {manager_model}")
    print(f"   Customer Insight: {customer_provider.upper()} - {customer_model}")
    
    # Tools 설정
    print("\nTools 설정 중...")
    
    # 데이터 로드
    print("   모든 보험 데이터 병렬 로드 중...")
    all_loader = AllInsuranceCSVLoader(data_dir="data")
    batches = all_loader.load_all_parallel(max_workers=2)
    
    if not batches:
        print("   경고: 로드된 데이터가 없습니다.")
        print("   Insurance Tools는 비활성화됩니다.")
        h2022_tools_enabled = False
    else:
        # 로드 결과 출력
        summary = all_loader.get_summary()
        print(f"   데이터 로드 완료:")
        for filename, info in summary['files'].items():
            print(f"      - {filename}: {info['records']:,}개 레코드")
        print(f"   총 {summary['total_records']:,}개 레코드")
        
        # 2022 데이터로 analyzer 설정 (기존 호환성 유지)
        batch_2022 = all_loader.get_batch('insurance_US_2022.csv')
        if batch_2022:
            analyzer = Insurance2022DataAnalyzer()
            analyzer.load_insurance_data(batch_2022)
            
            # 모든 H2022 Tools에 동일한 analyzer 주입
            demographic_tool = H2022DemographicTool(analyzer=analyzer)
            actuarial_tool = H2022ActuarialTool(analyzer=analyzer)
            risk_tool = H2022RiskTool(analyzer=analyzer)
            
            factory.register_tool(demographic_tool)
            factory.register_tool(actuarial_tool)
            factory.register_tool(risk_tool)
            print(f"   H2022 Tools 등록 (3개)")
            h2022_tools_enabled = True
        else:
            print("   경고: 2022 데이터를 찾을 수 없습니다.")
            print("   H2022 Tools는 비활성화됩니다.")
            h2022_tools_enabled = False
    
    # RAG Tools - Embedding 모델 선택
    print("\nEmbedding 모델 설정 중...")
    embedding_provider, embedding_model, embedding_dimension = select_embedding_model()
    
    # Embedding 모델 생성
    if embedding_provider == "openai":
        base_embedder = OpenAIEmbedder(model_name=embedding_model, dimension=embedding_dimension)
    elif embedding_provider == "ollama":
        base_embedder = OllamaEmbedder(model_name=embedding_model, dimension=embedding_dimension)
    else:
        raise ValueError(f"지원하지 않는 embedding provider: {embedding_provider}")
    
    embedder = H2022Embedding(embedder=base_embedder)
    vector_store = H2022VectorStore(dimension=embedding_dimension)
    retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=5)
    
    print(f"   Embedding 모델: {embedding_provider.upper()} - {embedding_model} (차원: {embedding_dimension})")
    
    multi_query_tool = MultiQueryRAGTool(retriever=retriever)
    hyde_tool = HyDERAGTool(retriever=retriever)
    compression_tool = ContextualCompressionRAGTool(retriever=retriever)
    self_query_tool = SelfQueryRAGTool(retriever=retriever)
    
    factory.register_tool(multi_query_tool)
    factory.register_tool(hyde_tool)
    factory.register_tool(compression_tool)
    factory.register_tool(self_query_tool)
    
    print(f"   RAG Tools 등록 (4개)")
    
    # Visualization Tool
    if h2022_tools_enabled:
        # analyzer를 사용하여 DataFrame이 있는 경우에만 등록
        viz_tool = VisualizationTool(dataframe=analyzer.df)
        factory.register_tool(viz_tool)
        print(f"   VisualizationTool 등록")
    else:
        print(f"   VisualizationTool 비활성화 (데이터 없음)")
    
    # Worker Agents 생성
    print("\nWorker Agents 생성 중...")
    
    # Factory에 등록된 모든 Tool 이름 자동 가져오기
    tool_names = factory.get_all_tool_names()
    
    customer_agent = factory.create_worker(
        name="customer_insight_agent",
        description="인구통계 프로파일링, 구매 행동 분석, 타겟 세그먼트 식별",
        tool_names=tool_names,
        llm_name="customer_insight_llm",
        worker_class=CustomerInsightAgent
    )
    
    print(f"   Customer Insight Agent 생성 (Tools: {len(tool_names)}개)")
    print(f"      등록된 Tools: {', '.join(tool_names)}")
    
    # Manager Agent 생성
    print("\nManager Agent 생성 중...")
    
    manager = factory.create_manager(
        name="insurance_manager",
        description="보험 데이터 분석 Manager Agent",
        worker_names=["customer_insight_agent"],
        llm_name="manager_llm",
        manager_class=InsuranceManagerAgent
    )
    
    print("   Manager Agent 생성 완료")
    
    return manager


async def chat_loop(manager):
    """대화 루프 """
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    
    print_section("대화 시작")
    print("Manager Agent와 대화를 시작합니다.")
    print("   종료하려면 'quit', 'exit', 'q'를 입력하세요.\n")
    
    # Chat History 초기화
    chat_history = InMemoryChatMessageHistory()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n종료합니다. 감사합니다!")
                break
            
            if not user_input:
                continue
            
            print("\nManager Agent: (처리 중...)")
            
            # Chat History에 사용자 메시지 추가
            chat_history.add_message(HumanMessage(content=user_input))
            
            # Context에 chat history 포함
            context = {
                'chat_history': chat_history.messages,
                'session_id': 'default'
            }
            
            # Manager Agent 실행
            result = await manager.process(user_input, context)
            
            if result.success:
                data = result.data
                answer = data.get('answer', '답변 없음')
                print(f"\n{answer}\n")
                
                # Chat History에 AI 응답 추가
                chat_history.add_message(AIMessage(content=answer))
                
                # 처리 통계
                if data.get('total_tasks'):
                    print(f"\n처리 통계:")
                    print(f"   총 작업: {data['total_tasks']}개")
                    print(f"   성공: {data.get('successful_tasks', 0)}개")
                    print(f"   대화 기록: {len(chat_history.messages)}개 메시지")
            else:
                print(f"\n오류: {result.error}")
        
        except KeyboardInterrupt:
            print("\n\n종료합니다. 감사합니다!")
            break
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
            logger.error(f"대화 오류: {str(e)}", exc_info=True)


async def main():
    """메인 함수"""
    try:
        print_header()
        
        # Agent 설정
        manager = await setup_agents()
        
        # 대화 시작
        await chat_loop(manager)
        
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        logger.error(f"메인 오류: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

