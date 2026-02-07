"""
Agent CLI
=========================
Author: Jin
Date: 2026.02.07
Version: 3.0 (Correct Tools Integration)

Description:
Manager Agent와 CLI로 상호작용하는 인터페이스입니다.
직접 구현한 ArxivSearchTool과 NewsSearchTool을 올바르게 통합했습니다.
"""
import asyncio
import sys
import argparse
import json
from typing import List, Dict, Any

# Factory & Agents
from factories.agent_factory import get_agent_factory
from agents.orchestrator.manager_agent import InsuranceManagerAgent
from agents.team.customer_insight.customer_insight_agent import CustomerInsightAgent
from agents.team.product.product_strategy_agent import ProductStrategyAgent

# Tools
from tools.insurance.h2022 import H2022DemographicTool, H2022ActuarialTool, H2022RiskTool
from tools.rag import MultiQueryRAGTool, HyDERAGTool, ContextualCompressionRAGTool, SelfQueryRAGTool
from tools.code_generator import VisualizationTool


from tools.data_collector.arxiv_tool import ArxivSearchTool
from tools.data_collector.news_search_tool import NewsSearchTool


# RAG Core
from rag.common.retriever import Retriever
from rag.insurance.h2022.h2022_embedding import H2022Embedding
from rag.insurance.h2022.h2022_vector_store import H2022VectorStore
from rag.common.openai_embedder import OpenAIEmbedder
from rag.common.ollama_embedder import OllamaEmbedder

# Utils
from base.agent.llm_base import LLMBase
from config.logging_config import logger
from utils.data.insurance_2022_data_analyzer import Insurance2022DataAnalyzer
from utils.data.all_insurance_csv_loader import AllInsuranceCSVLoader
from agents.llm_providers.llm_openai import LLMOpenAI
from agents.llm_providers.llm_ollama import LLMOllama
from utils.handler.llm_model_handler import LLMModelHandler
from utils.handler.embedding_model_handler import EmbeddingModelHandler


def print_header():
    print("\n" + "=" * 70)
    print("Insurance Data Analysis & Strategy Agent System")
    print("=" * 70 + "\n")


def print_section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")


def select_embedding_model() -> tuple:
    print("\nRAG를 위한 Embedding 모델 선택")
    print("   (Enter를 누르면 기본값 text-embedding-3-small 사용)\n")
    
    handler = EmbeddingModelHandler()
    choice = input("모델 목록을 보시겠습니까? (y/n, 또는 Enter로 기본값 사용): ").strip().lower()
    
    if not choice or choice == 'n':
        print("   text-embedding-3-small 선택됨")
        return ("openai", "text-embedding-3-small", 1536)
    
    result = handler.select_model()
    return result if result else ("openai", "text-embedding-3-small", 1536)


def create_llm_from_selection(provider: str, model: str, temperature: float = 0.7):
    if provider == "openai":
        return LLMOpenAI(model_name=model, temperature=temperature)
    elif provider == "ollama":
        return LLMOllama(model_name=model, temperature=temperature)
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")


def select_llm_with_handler(purpose: str) -> tuple:
    print(f"\n{purpose}을 위한 LLM 선택")
    print("   (Enter를 누르면 기본값 gpt-4o-mini 사용)\n")
    
    handler = LLMModelHandler()
    choice = input("모델 목록을 보시겠습니까? (y/n, 또는 Enter로 기본값 사용): ").strip().lower()
    
    if not choice or choice == 'n':
        print("   gpt-4o-mini 선택됨")
        return ("openai", "gpt-4o-mini")
    
    result = handler.select_model()
    return result if result else ("openai", "gpt-4o-mini")


async def setup_agents():
    """Agent 시스템 설정"""
    print_section("Agent 시스템 설정")
    
    # LLM 선택
    manager_provider, manager_model = select_llm_with_handler("Manager Agent")
    worker_provider, worker_model = select_llm_with_handler("Worker Agents")
    
    factory = get_agent_factory()
    
    # LLM 등록
    manager_llm = create_llm_from_selection(manager_provider, manager_model, temperature=0.5)
    worker_llm = create_llm_from_selection(worker_provider, worker_model, temperature=0.7)
    
    factory.register_llm("manager_llm", manager_llm)
    factory.register_llm("worker_llm", worker_llm)
    
    print(f"\nLLM 등록 완료")
    print(f"   Manager: {manager_model}")
    print(f"   Workers: {worker_model}")
    
    # Tools 설정
    print("\nTools 설정 중...")
    
    # 1. Insurance Data Tools
    print("   보험 데이터 로드 중...")
    all_loader = AllInsuranceCSVLoader(data_dir="data")
    batches = all_loader.load_all_parallel(max_workers=2)
    
    if batches:
        batch_2022 = all_loader.get_batch('insurance_US_2022.csv')
        if batch_2022:
            analyzer = Insurance2022DataAnalyzer()
            analyzer.load_insurance_data(batch_2022)
            
            factory.register_tool(H2022DemographicTool(analyzer=analyzer))
            factory.register_tool(H2022ActuarialTool(analyzer=analyzer))
            factory.register_tool(H2022RiskTool(analyzer=analyzer))
            factory.register_tool(VisualizationTool(dataframe=analyzer.df))
            print("   Insurance Analysis & Vis Tools 등록 완료")

    # 2. RAG Tools
    print("   RAG 시스템 초기화 중...")
    try:
        e_provider, e_model, e_dim = select_embedding_model()
        if e_provider == "openai":
            base_embedder = OpenAIEmbedder(model_name=e_model, dimension=e_dim)
        else:
            base_embedder = OllamaEmbedder(model_name=e_model, dimension=e_dim)
            
        embedder = H2022Embedding(embedder=base_embedder)
        vector_store = H2022VectorStore(dimension=e_dim)
        retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=5)
        
        factory.register_tool(MultiQueryRAGTool(retriever=retriever))
        factory.register_tool(HyDERAGTool(retriever=retriever))
        print("   RAG Tools 등록 완료")
    except Exception as e:
        logger.warning(f"RAG 설정 실패 Skipping: {e}")

    # 3. Data Collector Tools
    try:
        arxiv_tool = ArxivSearchTool()
        news_tool = NewsSearchTool()

        factory.register_tool(arxiv_tool)
        factory.register_tool(news_tool)
        print("   Data Collector Tools (Arxiv, News) 등록 완료")
    except Exception as e:
        logger.warning(f"데이터 수집 도구 설정 실패: {e}")

    # Agents 생성
    print("\nAgents 생성 중...")
    
    # 1. Customer Insight Agent
    insight_tools = ["h2022_demographic_analysis", "h2022_risk_analysis", "visualization_generator", "multi_query_rag"]
    valid_insight_tools = [t for t in insight_tools if t in factory._tools]
    
    factory.create_worker(
        name="customer_analyst",
        description="고객 데이터 분석 및 인사이트 도출 전문가",
        tool_names=valid_insight_tools,
        llm_name="worker_llm",
        worker_class=CustomerInsightAgent
    )
    
    # 2. Product Strategy Agent
    strategy_tools = ["arxiv_paper_search", "news_market_search"]
    valid_strategy_tools = [t for t in strategy_tools if t in factory._tools]
    
    factory.create_worker(
        name="product_planner",
        description="시장 조사 및 상품 전략 기획 전문가",
        tool_names=valid_strategy_tools,
        llm_name="worker_llm",
        worker_class=ProductStrategyAgent
    )
    
    print(f"   Worker Agents 생성 완료: customer_analyst, product_planner")
    
    # 3. Manager Agent
    manager = factory.create_manager(
        name="insurance_manager",
        description="보험 데이터 분석 및 기획 총괄 PM",
        worker_names=["customer_analyst", "product_planner"],
        llm_name="manager_llm",
        manager_class=InsuranceManagerAgent
    )
    
    print("   Manager Agent 생성 완료")
    return manager

def evaluate_with_ragas(query: str, answer: str, contexts: List[str]):
    """Ragas를 사용한 답변 품질 평가"""
    print_section("Ragas Quality Evaluation")
    
    if not contexts:
        print("   평가 불가: 검색된 증거 자료가 없습니다.")
        return

    print("   평가 진행 중... (LLM 및 임베딩 모델 호출)")
    
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # 1. 평가용 모델 설정
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        evaluator_embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={'trust_remote_code': True, 'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 데이터셋 생성
        data = {
            'question': [query],
            'answer': [answer],
            'contexts': [contexts]
        }
        dataset = Dataset.from_dict(data)

        # 3. 평가 실행
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        # 점수 추출 헬퍼 함수
        def safe_extract_score(metric_name, results_obj):
            try:
                # 1. 딕셔너리처럼 접근 시도
                score = results_obj[metric_name]
            except Exception:
                # 2. 접근 불가 시 Pandas로 변환하여 접근
                try:
                    df = results_obj.to_pandas()
                    score = df.iloc[0][metric_name]
                except Exception:
                    return 0.0

            # 3. 리스트인 경우 첫 번째 값 추출
            if isinstance(score, list):
                if len(score) > 0:
                    return float(score[0])
                return 0.0
            
            # 4. 이미 숫자인 경우
            try:
                return float(score)
            except (ValueError, TypeError):
                return 0.0

        # 안전하게 점수 추출
        faith_score = safe_extract_score('faithfulness', results)
        rel_score = safe_extract_score('answer_relevancy', results)

        print(f"\n   [평가 결과]")
        print(f"   - Faithfulness (사실 충실도): {faith_score:.4f}")
        print(f"   - Answer Relevancy (답변 관련성): {rel_score:.4f}")
        
        if faith_score < 0.7:
            print("\n   ⚠️ 경고: 답변의 충실도가 낮습니다. 환각 가능성이 있습니다.")

    except ImportError:
        print("   Error: 필수 패키지 로드 실패.")
        print("   pip install ragas langchain-huggingface sentence-transformers einops")
    except Exception as e:
        print(f"   [Warning] 평가 중 오류 발생: {e}")
        logger.warning(f"Ragas Evaluation Error: {e}", exc_info=True)



async def chat_loop(manager, enable_ragas: bool = False):
    """대화 루프"""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    
    print_section("시스템 준비 완료")
    print(f"   Ragas 평가 모드: {'ON' if enable_ragas else 'OFF'}")
    print("   종료하려면 quit, exit, q를 입력하세요.\n")
    
    chat_history = InMemoryChatMessageHistory()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n종료합니다. 감사합니다!")
                break
            
            if not user_input:
                continue
            
            print("\nManager Agent: 처리 중...")
            
            chat_history.add_message(HumanMessage(content=user_input))
            
            context = {
                'chat_history': chat_history.messages,
                'session_id': 'default'
            }
            
            result = await manager.process(user_input, context)
            
            if result.success:
                data = result.data
                answer = data.get('answer', '답변 없음')
                sources = data.get('sources', [])
                
                # 답변 출력
                print(f"\n{answer}\n")
                
                # 출처 출력
                if sources:
                    print("-" * 50)
                    print("참고 문헌 Sources:")
                    for idx, src in enumerate(sources, 1):
                        title = src.get('title', 'No Title')
                        url = src.get('url', 'N/A')
                        print(f"   {idx}. {title} ({url})")
                    print("-" * 50)
                
                chat_history.add_message(AIMessage(content=answer))
                
                # Ragas 평가
                if enable_ragas and manager.evidence:
                    evaluate_with_ragas(
                        query=user_input,
                        answer=manager.result,
                        contexts=manager.evidence
                    )
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
    parser = argparse.ArgumentParser(description="Insurance Agent System")
    parser.add_argument("--ragas", action="store_true", help="Enable Ragas evaluation")
    args = parser.parse_args()

    try:
        print_header()
        
        manager = await setup_agents()
        
        await chat_loop(manager, enable_ragas=args.ragas)
        
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        logger.error(f"메인 오류: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())