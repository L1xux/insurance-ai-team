"""
Product Strategy Agent
=========================
Author: Jin
Date: 2026.02.07
Version: 5.5

Description:
보험 상품 전략 수립을 담당하는 Worker Agent입니다.
입력 데이터(Context)의 길이에 따라 Summary 또는 Full Result를 선택적으로 사용합니다.
계획된 도구들을 병렬(asyncio.gather)로 실행합니다.
"""
from typing import Dict, Any, Optional, List, Tuple
import json
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from base.agent.agent_base import WorkerAgent
from base.agent.llm_base import LLMBase
from models.agent_model import AgentResult, AgentPlan
from config.logging_config import logger

class StrategySection(BaseModel):
    title: str = Field(description="전략 섹션 제목")
    content: str = Field(description="전략 내용 본문")
    cited_source_ids: List[int] = Field(
        description="본문에서 인용한 소스 ID 목록 (없으면 빈 리스트)",
        default_factory=list
    )

class StrategyReport(BaseModel):
    summary: str = Field(description="전체 전략의 핵심 요약 (3문장 이내)")
    sections: List[StrategySection] = Field(description="상세 전략 섹션 리스트")
    risk_assessment: str = Field(description="예상되는 리스크 및 대응 방안")

class ProductStrategyAgent(WorkerAgent):
    """보험 상품 기획 및 전략 수립 Worker Agent"""
    
    CONTEXT_LENGTH_LIMIT = 3000

    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        llm: Optional[LLMBase] = None
    ):
        super().__init__(name, description, tools, llm)
        
        if self.llm:
            self._setup_chains()
            
        logger.info(f"[{self.name}] 초기화 완료")

    def _setup_chains(self) -> None:
        """LLM Chain 및 Prompt 구성"""
        self.chat_model = self.llm.get_model()
        if not self.chat_model:
            logger.error(f"[{self.name}] 모델 로드 실패")
            return

        # 1. Planning Chain
        structured_plan_llm = self.chat_model.with_structured_output(
            AgentPlan,
            method="function_calling"
        )
        
        plan_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 논리적이고 데이터 기반의 보험 상품 기획자입니다.

제공된 고객 인사이트와 사용자 요청을 분석하여 상품 전략을 수립해야 합니다.
이를 위해 가설을 세우고, 외부 도구인 Arxiv와 News를 사용하여 근거를 확보하는 계획을 세우세요.

사용 가능한 도구:
{tools}

도구 활용 가이드:
1. arxiv_paper_search: 상품의 기술적 실현 가능성, 리스크 평가 모델 검증
2. news_market_search: 경쟁사 유사 상품, 최신 규제 이슈 파악

주의사항:
- 도구 사용 시 구체적인 키워드를 생성하세요.
- 불필요한 검색은 지양하고 핵심 근거 확보에 집중하세요.
"""),
            ("user", """요청 사항: {request}

[이전 분석 결과]
{context_data}""")
        ])
        
        self.plan_chain = plan_prompt | structured_plan_llm

        # 2. Synthesis Chain
        structured_report_llm = self.chat_model.with_structured_output(
            StrategyReport,
            method="function_calling"
        )

        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 수석 상품 기획자입니다.
            
내부 인사이트와 외부 검색 결과를 종합하여 신상품 전략 기획서를 작성하세요.
각 전략의 근거가 되는 자료(Source)를 반드시 명시해야 합니다.

작성 지침:
1. 제공된 [외부 증거 자료] 목록의 ID를 참고하세요.
2. 전략 내용 작성 시, 반드시 근거가 되는 소스 ID를 cited_source_ids 필드에 포함하세요.
3. 근거가 없는 내용은 추측하여 작성하지 마세요.
"""),
            ("user", """요청: {request}

[내부 인사이트]
{internal_context}

[외부 증거 자료]
{evidence_context}""")
        ])
        
        self.synthesis_chain = synthesis_prompt | structured_report_llm

    async def process(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """상품 전략 수립 프로세스 실행"""
        try:
            logger.info(f"[{self.name}] 기획 프로세스 시작: {message[:50]}...")
            
            # 1. Context 데이터 처리
            previous_results = context.get('previous_results', []) if context else []
            internal_context = self._optimize_context(previous_results)
            
            # 2. Plan 수립
            tool_schemas = [tool.get_schema().to_dict() for tool in self.tools.values()]
            
            plan = await self.plan_chain.ainvoke({
                "tools": json.dumps(tool_schemas, indent=2, ensure_ascii=False),
                "request": message,
                "context_data": internal_context
            })
            
            logger.info(f"[{self.name}] Plan 수립: {plan.reasoning}")
            
            # 3. Tool 실행 (병렬 처리)
            tool_results = []
            if plan.tools:
                tool_results = await self._execute_tools(plan.tools)
            
            # Fail-Fast
            if not tool_results or all(not r.get('success', False) for r in tool_results):
                logger.warning(f"[{self.name}] 외부 근거 확보 실패 또는 결과 없음")
                return AgentResult(
                    success=False,
                    data={"error": "외부 근거 데이터 확보에 실패"},
                    error="Insufficient evidence data"
                )

            # 4. 결과 종합 및 전략 수립
            formatted_evidence, source_map = self._format_evidence_for_llm(tool_results)
            
            strategy_report = await self.synthesis_chain.ainvoke({
                "request": message,
                "internal_context": internal_context,
                "evidence_context": formatted_evidence
            })
            
            # 5. 최종 결과 구성
            final_output = self._construct_final_output(strategy_report, source_map, formatted_evidence)
            
            return AgentResult(
                success=True,
                data=final_output
            )

        except Exception as e:
            logger.error(f"[{self.name}] 프로세스 실패: {str(e)}")
            return AgentResult(success=False, data=None, error=str(e))

    async def _execute_tools(self, tool_plans: List) -> List[Dict[str, Any]]:
        """Tool 병렬 실행"""
        
        async def run_single_tool(plan) -> Dict[str, Any]:
            tool_name = plan.tool_name
            
            # [Fix] 입력값 전처리 
            raw_input = plan.query
            
            # 1. None 체크: 쿼리가 없으면 빈 JSON으로 처리
            if raw_input is None:
                logger.warning(f"[{self.name}] {tool_name} 쿼리가 None입니다. 빈 요청으로 처리합니다.")
                tool_input = "{}"
                
            # 2. Dict 체크: 이미 객체로 들어왔다면 JSON 문자열로 변환
            elif isinstance(raw_input, dict):
                tool_input = json.dumps(raw_input, ensure_ascii=False)
            
            # 3. 그 외: 문자열로 강제 변환
            else:
                tool_input = str(raw_input)

            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning(f"[{self.name}] Tool 없음: {tool_name}")
                return {
                    'tool_name': tool_name,
                    'success': False,
                    'error': "Tool not found"
                }
            
            logger.info(f"[{self.name}] Tool 실행 시작: {tool_name} (Input: {tool_input[:50]}...)")
            
            try:
                # BaseTool.ainvoke는 문자열 또는 Dict를 받음
                if hasattr(tool, 'ainvoke'):
                    tool_result_str = await tool.ainvoke(tool_input)
                    
                    # 결과 파싱 시도
                    try:
                        # 이미 Dict라면 그대로 사용
                        if isinstance(tool_result_str, dict):
                            tool_result = tool_result_str
                        else:
                            tool_result = json.loads(tool_result_str)
                    except (json.JSONDecodeError, TypeError):
                        # JSON 파싱 실패 시 원본 문자열을 data로 포장
                        tool_result = {"success": True, "data": str(tool_result_str)}
                        
                    return {
                        'tool_name': tool_name,
                        'success': tool_result.get('success', True),
                        'data': tool_result,
                        'error': tool_result.get('error')
                    }
                else:
                    return {
                        'tool_name': tool_name,
                        'success': False,
                        'error': "Tool execution method not supported"
                    }
            except Exception as e:
                logger.error(f"[{self.name}] Tool 실행 오류: {str(e)}")
                return {
                    'tool_name': tool_name,
                    'success': False,
                    'error': str(e)
                }

        # 모든 도구 실행 작업을 Task로 생성하여 병렬 실행
        tasks = [run_single_tool(plan) for plan in tool_plans]
        results = await asyncio.gather(*tasks)
        
        return results

    def _optimize_context(self, previous_results: List[Dict[str, Any]]) -> str:
        """이전 결과의 길이에 따라 Summary 또는 Full Result 선택"""
        if not previous_results:
            return "이전 분석 결과 없음"

        full_text_list = []
        for res in previous_results:
            full_res = res.get('output', '') or res.get('full_result', '') or res.get('answer', '')
            if isinstance(full_res, dict):
                full_text_list.append(json.dumps(full_res, ensure_ascii=False))
            else:
                full_text_list.append(str(full_res))
        
        full_context = "\n".join(full_text_list)
        
        if len(full_context) > self.CONTEXT_LENGTH_LIMIT:
            logger.info(f"[{self.name}] Context 길이({len(full_context)}) 초과 -> Summary 사용")
            summary_list = []
            for res in previous_results:
                summary_list.append(str(res.get('summary', '')))
            return "\n".join(summary_list)
            
        return full_context

    def _format_evidence_for_llm(self, tool_results: List[Dict[str, Any]]) -> Tuple[str, Dict[int, Any]]:
        """도구 실행 결과를 LLM이 인용 가능한 포맷으로 변환"""
        formatted_lines = []
        source_map = {}
        source_id = 1
        
        for res in tool_results:
            if not res.get('success', False):
                continue
                
            tool_data = res.get('data', {})
            
            # Arxiv 논문
            if 'papers' in tool_data:
                for paper in tool_data['papers']:
                    title = paper.get('title', 'No Title')
                    summary = paper.get('summary', '')[:200]
                    line = f"[Source {source_id}] (Arxiv) {title} - {summary}..."
                    formatted_lines.append(line)
                    source_map[source_id] = {"type": "arxiv", "title": title, "url": paper.get('url')}
                    source_id += 1
            
            # News 기사
            elif 'articles' in tool_data:
                for article in tool_data['articles']:
                    title = article.get('title', 'No Title')
                    snippet = article.get('description', '')[:200]
                    source = article.get('source', 'Unknown')
                    line = f"[Source {source_id}] (News) [{source}] {title} - {snippet}..."
                    formatted_lines.append(line)
                    source_map[source_id] = {"type": "news", "title": title, "url": article.get('url'), "source": source}
                    source_id += 1
                    
        return "\n".join(formatted_lines), source_map

    def _construct_final_output(self, report: StrategyReport, source_map: Dict[int, Any], raw_evidence: str) -> Dict[str, Any]:
        """최종 결과 데이터 구조화"""
        
        full_result_lines = [f"# {report.summary}\n"]
        used_sources = set()
        
        for section in report.sections:
            full_result_lines.append(f"## {section.title}")
            full_result_lines.append(section.content)
            
            if section.cited_source_ids:
                citations = []
                for sid in section.cited_source_ids:
                    if sid in source_map:
                        src = source_map[sid]
                        citations.append(f"- {src['title']} ({src.get('url', '')})")
                        used_sources.add(sid)
                
                if citations:
                    full_result_lines.append("\n**[참고 자료]**")
                    full_result_lines.extend(citations)
            full_result_lines.append("")
            
        full_result_lines.append(f"## 리스크 평가\n{report.risk_assessment}")
        
        sources_list = []
        for sid in used_sources:
            sources_list.append(source_map[sid])
            
        return {
            "summary": report.summary,
            "full_result": "\n".join(full_result_lines),
            "sources": sources_list,
            "evidence_data": raw_evidence
        }