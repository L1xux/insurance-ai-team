# Insurance Data Analysis Agent System

LLM 기반 Multi-Agent 시스템을 활용한 보험 데이터 분석 및 RAG(Retrieval-Augmented Generation) 플랫폼

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://www.langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 개요

Multi-Agent 아키텍처와 RAG 기술을 결합하여 보험 데이터를 분석하는 시스템입니다. Manager-Worker 패턴을 통해 복잡한 분석 작업을 처리하며, 다양한 LLM 및 임베딩 모델을 지원합니다.

### 아키텍처 특징

- **Multi-Agent 시스템**: Manager Agent가 Worker Agent들을 조율하여 작업 분배 및 결과 통합
- **RAG 기반 문서 검색**: PDF 문서를 벡터화하여 의미 기반 검색 및 컨텍스트 기반 답변 생성
- **의존성 주입 패턴**: 확장 가능하고 테스트 가능한 아키텍처
- **병렬 데이터 처리**: ThreadPoolExecutor를 활용한 동시 데이터 로딩
- **도메인 특화 분석**: MEPS 보험 데이터에 특화된 인구통계, 리스크, 계리 분석

## 주요 기능

### Multi-Agent 시스템

- **Manager Agent**: 사용자 요청 분석, 작업 계획 수립, Worker Agent 조율, 결과 통합
- **Worker Agents**:
  - Customer Insight Agent: 인구통계 및 구매 행동 분석
  - Actuarial Agent: 청구액 및 준비금 계산

### RAG (Retrieval-Augmented Generation)

- **문서 인덱싱**: PDF 문서를 청크 단위로 분할하여 벡터화 및 PostgreSQL 저장
- **의미 기반 검색**: pgvector를 활용한 KNN 검색
- **검색 전략**:
  - Multi-Query RAG: 단일 쿼리를 여러 관점으로 확장
  - HyDE: 가상 문서 생성 기반 검색
  - Contextual Compression: 관련성 높은 컨텍스트만 추출
  - Self-Query: 구조화된 메타데이터 기반 필터링

### 데이터 분석 도구

- 인구통계 분석: 연령, 성별, 지역별 보험 가입 패턴
- 리스크 분석: 건강 상태, 흡연 여부 등 리스크 요인 평가
- 계리 분석: 보험료 산정, 청구액 예측, 준비금 계산
- 자동 시각화: LLM 기반 시각화 코드 생성 및 실행

### 모델 지원

- **LLM**: OpenAI (GPT-4, GPT-3.5), Ollama (Llama, Mistral 등)
- **Embedding**: OpenAI Embedding (text-embedding-3), Ollama Embedding (nomic-embed-text 등)
- 런타임 모델 선택 및 교체 지원

## 아키텍처

### 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│                    (CLI / API)                          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Manager Agent                               │
│  - 요청 분석 및 계획 수립                                │
│  - Worker Agent 조율                                     │
│  - 결과 통합                                             │
└────────────┬───────────────────────┬────────────────────┘
             │                       │
    ┌────────▼────────┐     ┌────────▼────────┐
    │ Customer       │     │ Actuarial      │
    │ Insight Agent  │     │ Agent          │
    └────────┬───────┘     └────────┬───────┘
             │                       │
    ┌────────▼───────────────────────▼────────┐
    │         Tools Layer                      │
    │  - Demographic Analysis                  │
    │  - Risk Analysis                        │
    │  - RAG Tools                            │
    │  - Visualization                        │
    └────────┬────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────┐
    │      Data & RAG Layer                  │
    │  - Data Loaders (병렬 처리)             │
    │  - Vector Store (PostgreSQL + pgvector)│
    │  - Embedding Models                    │
    └────────────────────────────────────────┘
```

### 디자인 패턴

- **Factory Pattern**: Agent와 Tool 생성 및 관리
- **Dependency Injection**: Embedding 모델, LLM 모델 주입
- **Strategy Pattern**: 다양한 RAG 검색 전략 구현
- **Template Method Pattern**: Agent 처리 파이프라인 정의

## 기술 스택

### Core Technologies
- Python 3.10+
- LangChain: LLM 체인 구성 및 Agent 프레임워크
- PostgreSQL + pgvector: 벡터 데이터베이스
- asyncio: 비동기 처리

### LLM & Embedding
- OpenAI API: GPT-4, GPT-3.5, text-embedding-3
- Ollama: 로컬 LLM 및 임베딩 모델

### Data Processing
- pandas: 데이터 분석 및 처리
- numpy: 수치 연산
- matplotlib/seaborn: 데이터 시각화

### Infrastructure
- Docker: 컨테이너화
- Kubernetes: 오케스트레이션
- Jenkins: CI/CD 파이프라인

## 프로젝트 구조

```
llm-data-analyzer/
├── agents/                    # Agent 구현
│   ├── orchestrator/         # Manager Agent
│   ├── team/                 # Worker Agents
│   │   ├── customer_insight/
│   │   └── actuarial/
│   └── llm_providers/        # LLM 구현체
│
├── base/                     # 베이스 클래스
│   ├── agent/                # Agent 인터페이스
│   ├── rag/                  # RAG 베이스 클래스
│   └── data_loader_base.py
│
├── factories/                 # Factory 패턴
│   ├── agent_factory.py      # Agent 생성 관리
│   └── llm_factory.py
│
├── rag/                      # RAG 시스템
│   ├── common/               # 공통 RAG 컴포넌트
│   └── insurance/            # 보험 도메인 RAG
│
├── tools/                    # 분석 도구
│   ├── insurance/            # 보험 분석 도구
│   ├── rag/                  # RAG 검색 도구
│   └── code_generator/       # 시각화 생성
│
├── utils/                    # 유틸리티
│   ├── data/                 # 데이터 로더/분석기
│   └── handler/              # 모델 선택 핸들러
│
├── models/                   # 데이터 모델
├── database/                 # DB 연결 관리
├── config/                   # 설정 파일
└── main.py                   # 메인 진입점
```

## 시작하기

### 사전 요구사항

- Python 3.10 이상
- PostgreSQL 15 이상 (pgvector 확장 필요)
- OpenAI API 키 또는 Ollama (로컬 실행 시)

### 설치

1. 저장소 클론
```bash
git clone <repository-url>
cd llm-data-analyzer
```

2. 가상 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp env.properties.example env.properties
# env.properties 파일에 API 키 및 DB 설정 추가
```

5. 데이터베이스 설정
```bash
# PostgreSQL에 pgvector 확장 설치
psql -U postgres -c "CREATE EXTENSION vector;"
```

### 실행

```bash
python main.py
```

## 핵심 구현

### 의존성 주입 패턴

임베딩 모델을 런타임에 주입하여 유연성 확보:

```python
# Embedding 모델 선택 및 주입
embedding_provider, embedding_model, dimension = select_embedding_model()

if embedding_provider == "openai":
    base_embedder = OpenAIEmbedder(model_name=embedding_model, dimension=dimension)
elif embedding_provider == "ollama":
    base_embedder = OllamaEmbedder(model_name=embedding_model, dimension=dimension)

# H2022Embedding에 주입
embedder = H2022Embedding(embedder=base_embedder)
```

### 병렬 데이터 로딩

ThreadPoolExecutor를 활용한 동시 처리:

```python
all_loader = AllInsuranceCSVLoader(data_dir="data")
batches = all_loader.load_all_parallel(max_workers=2)
```

### Factory 패턴을 통한 Tool 관리

```python
# Tool 등록
factory.register_tool(demographic_tool)
factory.register_tool(actuarial_tool)

# 등록된 모든 Tool 자동 가져오기
tool_names = factory.get_all_tool_names()
```

### RAG 파이프라인

```python
# 문서 인덱싱
indexer = H2022Indexer(embedder=embedder, vector_store=vector_store)
indexer.index_document("data/h2022doc.pdf")

# 검색
retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=5)
results = retriever.search("보험료 영향 요인")
```

## 기능 상세

### 데이터 분석

- 인구통계 분석: 연령대별, 성별, 지역별 보험 가입 현황
- 리스크 분석: 건강 상태, 흡연 여부 등 리스크 요인 평가
- 계리 분석: 보험료 산정, 청구액 예측, 준비금 계산

### RAG 검색

- Multi-Query RAG: 단일 쿼리를 여러 관점으로 확장하여 검색
- HyDE: 가상 문서 생성 후 검색으로 정확도 향상
- Contextual Compression: 관련성 높은 컨텍스트만 추출
- Self-Query: 구조화된 메타데이터 기반 필터링 검색

### 자동 시각화

LLM이 데이터 특성을 분석하여 시각화 코드 생성:

```python
viz_tool = VisualizationTool(dataframe=analyzer.df)
result = viz_tool.invoke("연령대별 보험료 분포를 시각화해주세요")
```

## 설정

### 환경 변수 (env.properties)

```properties
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=insurance_db
DB_USER=postgres
DB_PASSWORD=your_password

# Ollama (선택사항)
OLLAMA_BASE_URL=http://localhost:11434
```

## 성능 최적화

- 병렬 데이터 로딩: ThreadPoolExecutor를 활용한 동시 처리
- 벡터 인덱싱: pgvector의 HNSW 인덱스로 빠른 검색
- 비동기 처리: asyncio를 활용한 Agent 간 비동기 통신
- 배치 임베딩: 여러 텍스트를 한 번에 임베딩하여 API 호출 최소화

## 테스트

```bash
# 타입 체크
mypy .

# 린팅
flake8 .

# 포맷팅
black .
isort .
```

## 배포

### Docker

```bash
docker build -t llm-data-analyzer .
docker run -p 8000:8000 llm-data-analyzer
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

## 기술적 특징

### 확장 가능한 아키텍처

- 베이스 클래스 기반 설계: 새로운 Agent나 Tool 추가 용이
- 인터페이스 분리: 각 컴포넌트의 명확한 책임 분리
- Factory 패턴: 컴포넌트 생성 및 관리 중앙화

### 유연한 모델 지원

- Provider 추상화: OpenAI, Ollama 등 다양한 제공자 지원
- 런타임 모델 선택: 사용자가 원하는 모델을 선택 가능
- Handler 패턴: 모델 선택 로직 캡슐화

### 도메인 특화 설계

- 보험 도메인 모델: MEPS 데이터 구조에 특화된 모델 정의
- 전문 분석 도구: 인구통계, 리스크, 계리 분석 전용 도구
- 메타데이터 활용: 문서 구조 정보를 검색에 활용

### 프로덕션 고려사항

- 로깅 시스템: 구조화된 로깅으로 디버깅 용이
- 에러 핸들링: 예외 상황에 대한 안전한 처리
- CI/CD: Jenkins를 통한 자동화된 배포
- 타입 안정성: Python 타입 힌트 및 MyPy 활용

## 라이선스

MIT License

## 작성자

Jin

- 프로젝트 기간: 2025.09 - 2025.10
- 버전: 2.0

## 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [MEPS Data Documentation](https://meps.ahrq.gov/)

---