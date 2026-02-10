# Insurance AI Team

보험 데이터 분석 및 상품 전략 수립을 위한 Multi-Agent AI 시스템입니다.

## 기술

### 핵심 프레임워크
- **Python 3.11+**
- **LangChain**
- **OpenAI GPT-4o / GPT-4o-mini**
- **Ollama** (로컬 LLM 지원)

### 데이터 & 임베딩
- **PostgreSQL + pgvector** - 프로덕션급 벡터 데이터베이스
- **OpenAI Embeddings** - text-embedding-3-small/large
- **Ollama Embeddings** - nomic-embed-text, mxbai-embed-large
- **ArXiv API** - 학술 논문 검색
- **GNews API** - 뉴스 기사 크롤링

### 검증 & 출력
- **Ragas** - LLM 품질 평가 (Faithfulness, Relevancy)
- **Pydantic** - 스키마 검증 및 타입 안정성
- **matplotlib / seaborn** - 데이터 시각화

### 인프라
- **Docker** - 컨테이너화
- **Kubernetes** - 오케스트레이션
- **Jenkins** - CI/CD 파이프라인

## 워크플로우

```
사용자 입력
    ↓
┌─────────────────────┐
│ Manager Agent       │  요청 분석 및 작업 계획 수립 (Planning)
│ (Planning)          │  - ExecutionPlan 생성 (Pydantic 검증)
└─────────────────────┘  - Worker Agent 선택 및 할당
    ↓
    ├──────────────────┬──────────────────┐
    ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  (병렬 실행 가능)
│ Customer     │  │ Product      │
│ Insight      │  │ Strategy     │
│ Agent        │  │ Agent        │
└──────────────┘  └──────────────┘
    │                  │
    │ Tool 자율 선택   │ Tool 자율 선택
    │ (ReAct)          │ (ReAct)
    ↓                  ↓
 ┌─────────┐        ┌─────────┐
 │ RAG     │        │ Arxiv   │
 │ Search  │        │ Search  │
 ├─────────┤        ├─────────┤
 │ Demo    │        │ News    │
 │ Analysis│        │ Crawl   │
 ├─────────┤        └─────────┘
 │ Risk    │
 │ Analysis│
 ├─────────┤
 │ Visual  │
 │ ization │
 └─────────┘
    │                  │
    └────────┬─────────┘
             ↓
┌─────────────────────┐
│ Manager Agent       │  결과 통합 및 최종 답변 생성
│ (Aggregation)       │  - Inter-Agent Context Sharing
└─────────────────────┘  - Sources & Evidence 수집
    ↓
사용자 출력
    ↓
(Optional) Ragas 평가
```

**Inter-Agent Context Sharing**
- Manager가 각 Worker 결과를 accumulated_context에 저장
- 다음 Worker에게 이전 결과 전달 (summary + full_result)
- Dynamic Context Optimization: 3000자 기준 자동 전환

## 에이전트 및 LLM

### 1. Manager Agent
**역할**: 총괄 관리자 (PM)

사용자 요청을 받아 어떤 작업이 필요한지 분석하고, 적합한 전문가(Worker Agent)를 선택하여 작업을 할당합니다.
- 사용자 질문 분석 및 작업 계획 수립
- 적합한 전문가 Agent 선택 및 순차 실행
- 각 전문가의 결과를 통합하여 최종 답변 생성
- 참고 문헌 및 데이터 출처 수집

### 2. Customer Insight Agent
**역할**: 고객 데이터 분석 전문가

보험 가입자 데이터를 분석하여 인사이트를 도출합니다. 필요에 따라 아래 도구들을 자동으로 선택하여 사용합니다.

**사용 가능한 도구:**
- **인구통계 분석**: 나이, 성별, 지역별 가입자 분포 분석
- **리스크 분석**: 고위험 고객 그룹 식별 및 특성 분석
- **시각화 생성**: 차트 및 그래프 자동 생성
- **문서 검색**: 과거 보험 문서에서 관련 정보 검색

### 3. Product Strategy Agent
**역할**: 상품 기획 및 시장 조사 전문가

외부 데이터(학술 논문, 뉴스)를 수집하여 시장 동향을 분석하고 상품 전략을 수립합니다.

**사용 가능한 도구:**
- **학술 논문 검색**: arXiv에서 최신 연구 논문 검색
- **뉴스 검색**: 글로벌 보험/헬스케어 뉴스 수집

수집된 데이터를 바탕으로 전략 보고서를 작성하며, 각 전략마다 출처를 명시합니다.

### 4. 문서 검색 시스템 (RAG)
**역할**: 과거 문서 기반 정보 검색

PostgreSQL 데이터베이스에 저장된 보험 문서를 검색합니다.
- 사용자 질문을 여러 관점으로 확장하여 검색 (Multi-Query RAG)
- 페이지, 섹션, 카테고리별 필터링 가능
- 검색 결과의 출처(페이지 번호) 자동 추적

### 5. 시각화 생성 도구
**역할**: 데이터 차트 자동 생성

사용자의 자연어 요청을 받아 Python 코드를 자동으로 생성하고 실행하여 차트를 만듭니다.
- "나이별 분포를 막대 그래프로 그려줘" → 자동으로 코드 생성 및 실행
- matplotlib/seaborn 라이브러리 사용
- 생성된 이미지는 자동으로 저장

### 6. 품질 평가 시스템 (Optional)
**역할**: 답변 품질 검증

Ragas 프레임워크를 사용하여 AI 답변의 품질을 측정합니다.
- **충실도**: 답변이 검색된 자료에 근거하는가?
- **관련성**: 답변이 사용자 질문과 관련있는가?
- `--ragas` 옵션으로 활성화

## 프로젝트 구조

```
insurance-ai-team/
├── agents/                                  # Agent 구현
│   ├── orchestrator/
│   │   └── manager_agent.py                 # Manager Agent (Planning & Orchestration)
│   ├── team/
│   │   ├── customer_insight/
│   │   │   └── customer_insight_agent.py    # 고객 인사이트 분석 Worker
│   │   └── product/
│   │       └── product_strategy_agent.py    # 상품 전략 수립 Worker
│   └── llm_providers/
│       ├── llm_openai.py                    # OpenAI LLM Wrapper
│       └── llm_ollama.py                    # Ollama LLM Wrapper
├── base/                                    # 베이스 클래스
│   ├── agent/
│   │   ├── agent_base.py                    # AgentBase, ManagerAgent, WorkerAgent
│   │   ├── llm_base.py                      # LLMBase 인터페이스
│   │   └── data_analyzer_base.py            # DataAnalyzerBase
│   └── rag/
│       ├── retriever_base.py                # RetrieverBase
│       ├── vector_store_base.py             # VectorStoreBase
│       ├── embedding_base.py                # EmbeddingBase
│       └── preprocessor_base.py             # PreprocessorBase
├── tools/                                   # LangChain BaseTool 구현
│   ├── insurance/h2022/
│   │   ├── h2022_demographic_tool.py        # 인구통계 분석
│   │   ├── h2022_actuarial_tool.py          # 보험료 통계 분석
│   │   └── h2022_risk_tool.py               # 고위험군 식별
│   ├── code_generator/
│   │   └── visualization_tool.py            # LLM 기반 시각화 코드 생성
│   ├── data_collector/
│   │   ├── arxiv_tool.py                    # ArXiv 논문 검색
│   │   └── news_search_tool.py              # GNews 뉴스 크롤링
│   └── rag/
│       ├── multi_query.py                   # Multi-Query RAG
│       ├── hyde.py                          # HyDE RAG
│       ├── self_query.py                    # Self-Query RAG
│       └── contextual_compression.py        # Contextual Compression RAG
├── rag/                                     # RAG 파이프라인
│   ├── common/
│   │   ├── retriever.py                     # 기본 Retriever 구현
│   │   ├── openai_embedder.py               # OpenAI Embedder
│   │   └── ollama_embedder.py               # Ollama Embedder
│   └── insurance/h2022/
│       ├── h2022_vector_store.py            # PostgreSQL + pgvector
│       ├── h2022_embedding.py               # H2022 Embedding
│       ├── h2022_preprocessor.py            # H2022 문서 전처리
│       ├── h2022_document_loader.py         # PDF 문서 로더
│       └── h2022_indexer.py                 # 벡터 인덱싱
├── factories/                               # Factory Pattern
│   ├── agent_factory.py                     # Agent 생성 Factory
│   └── llm_factory.py                       # LLM 생성 Factory
├── models/                                  # Pydantic 모델
│   ├── agent_model.py                       # AgentResult, ExecutionPlan, AgentPlan
│   ├── tool_model.py                        # ToolSchema
│   ├── insurance_model.py                   # Insurance 데이터 모델
│   └── h2022_document_model.py              # H2022Document 스키마
├── utils/                                   # 유틸리티
│   ├── data/
│   │   ├── insurance_2022_data_analyzer.py  # 2022 데이터 분석 엔진
│   │   ├── insurance_2022_data_loader.py    # CSV 로더
│   │   └── all_insurance_csv_loader.py      # 병렬 CSV 로더
│   ├── handler/
│   │   ├── llm_model_handler.py             # LLM 선택 CLI 핸들러
│   │   └── embedding_model_handler.py       # Embedding 선택 CLI 핸들러
│   └── code_processor.py                    # 코드 샌드박스 실행
├── config/                                  # 설정
│   ├── settings.py                          # 전역 설정
│   └── logging_config.py                    # 로깅 설정
├── database/
│   └── db_connection.py                     # PostgreSQL 연결 관리
├── scripts/
│   └── create_vectordb_for_h2022_.py        # 벡터 DB 초기화 스크립트
├── data/                                    # 데이터 디렉토리
│   ├── insurance_US_2022.csv                # 보험 데이터 (2022)
│   ├── insurance_US_2023.csv                # 보험 데이터 (2023)
│   ├── h2022doc.pdf                         # H2022 참조 문서
│   └── visualizations/                      # 생성된 차트 이미지
├── k8s/                                     # Kubernetes 매니페스트
│   ├── deploy.yaml                          # Deployment
│   ├── service.yaml                         # Service
│   ├── ingress.yaml                         # Ingress
│   └── secret.yaml                          # Secret
├── Dockerfile                               # Docker 이미지 빌드
├── Jenkinsfile                              # CI/CD 파이프라인
├── requirements.txt                         # Python 의존성
├── main.py                                  # 메인 CLI 인터페이스
└── README.md                                # 프로젝트 문서
```

## 사용법

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd insurance-ai-team

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 설정

1. 환경 변수 설정 (`.env` 파일):
```bash
# OpenAI API 키
OPENAI_API_KEY=your_api_key_here

# PostgreSQL 설정
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=insurance_vector_db
```

2. PostgreSQL + pgvector 설치 및 초기화:
```bash
# Docker로 PostgreSQL + pgvector 실행
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=insurance_vector_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 벡터 DB 초기화 (H2022 문서 임베딩)
python scripts/create_vectordb_for_h2022_.py
```

### 시스템 실행

#### 대화형 모드
```bash
python main.py
```

실행 시 LLM 및 Embedding 모델 선택:
- Manager Agent용 LLM 선택 (기본값: gpt-4o-mini)
- Worker Agent용 LLM 선택 (기본값: gpt-4o-mini)
- Embedding 모델 선택 (기본값: text-embedding-3-small)

#### Ragas 평가 활성화
```bash
python main.py --ragas
```

### 시스템 흐름

1. **모델 선택**: LLM 및 Embedding 모델 선택 (OpenAI / Ollama)
2. **Tools 초기화**: 보험 데이터 로드 및 RAG 시스템 초기화
3. **Agents 생성**: Manager 및 Worker Agent 생성
4. **대화 시작**: 사용자 질문 입력
5. **Manager Planning**: 요청 분석 및 Worker 선택
6. **Worker 실행**: Tool 자율 선택 및 병렬 실행
7. **결과 통합**: Manager가 최종 답변 생성
8. **출력**: 답변 + 참고 문헌 Sources 표시
9. **(Optional) Ragas 평가**: Faithfulness & Answer Relevancy 측정

### 출력 파일

- **시각화 이미지**: `data/visualizations/visualization_{timestamp}.png`
- **로그**: 콘솔 및 로거를 통해 실시간 출력

## Docker 배포

### 이미지 빌드
```bash
docker build -t insurance-ai-team:latest .
```

### 컨테이너 실행
```bash
docker run -d \
  --name insurance-ai \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e POSTGRES_HOST=postgres-pgvector \
  --link postgres-pgvector:postgres \
  -p 8000:8000 \
  insurance-ai-team:latest
```

## Kubernetes 배포

```bash
# Secret 생성
kubectl apply -f k8s/secret.yaml

# Deployment & Service
kubectl apply -f k8s/deploy.yaml
kubectl apply -f k8s/service.yaml

# Ingress (Optional)
kubectl apply -f k8s/ingress.yaml

# 상태 확인
kubectl get pods
kubectl logs -f <pod-name>
```

## CI/CD (Jenkins)

Jenkins 파이프라인이 자동으로 실행:
1. 코드 체크아웃
2. Python 의존성 설치
3. 테스트 실행
4. Docker 이미지 빌드
5. Kubernetes 배포

## 개발

### 타입 체크
```bash
mypy .
```

### 코드 포맷팅
```bash
black .
isort .
```

### 테스트 실행
```bash
pytest tests/
```

