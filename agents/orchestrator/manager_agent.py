"""
Manager Agent - Orchestrator
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
Worker Agent들을 orchestrate하는 Manager Agent입니다.
LCEL을 사용하여 plan 수립 및 실행을 관리합니다.
"""
from typing import Dict, Any, Optional, List

from langchain_core.prompts import ChatPromptTemplate

from base.agent.agent_base import ManagerAgent, WorkerAgent
from base.agent.llm_base import LLMBase
from models.agent_model import AgentResult, ExecutionPlan
from config.logging_config import logger

class InsuranceManagerAgent(ManagerAgent):
    """보험 데이터 분석 Manager Agent"""
    
    def __init__(
        self,
        name: str,
        description: str,
        worker_agents: List[WorkerAgent],
        llm: LLMBase
    ):
        """
        Manager Agent 초기화
        
        Args:
            name: Agent 이름
            description: Agent 설명
            worker_agents: Worker Agent 리스트
            llm: LLM 인스턴스
        """
        super().__init__(name, description, worker_agents, llm) 
        
        # Chain 구성
        self._setup_chains()
        
        logger.info(f"[{self.name}] Manager Agent 초기화 완료")
    
    def _setup_chains(self) -> None:
        """체인 구성"""
        
        # LLMBase 구현체로부터 모델 가져오기
        self.chat_model = self.llm.get_model()
        if not self.chat_model:
            logger.warning(f"[{self.name}] 모델 없음 - Chain 구성 불가")
            return
        
        # Worker Agent 정보
        worker_info = self._get_worker_info_text()
        
        # Structured output을 위한 Chat Model 설정 (function_calling 모드 사용)
        structured_llm = self.chat_model.with_structured_output(
            ExecutionPlan,
            method="function_calling"  # 더 유연한 function calling 모드
        )
        
        # Planning Chain
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 보험 데이터 분석 작업을 계획하는 Manager Agent입니다.

사용 가능한 Worker Agents:
{worker_info}

작업 계획 수립 지침:
1. 복잡한 분석 요청은 가능한 한 하나의 작업으로 통합하세요
2. 각 Worker Agent는 여러 도구를 순차적으로 사용할 수 있습니다
3. 데이터 분석 + RAG 분석이 필요한 경우, 하나의 작업으로 요청하세요
4. 작업 설명(description)에 구체적인 요구사항을 모두 포함하세요
   예: "데이터 분석 후 RAG로 해석하고 인사이트 제공"

ExecutionPlan 형식:
- analysis: 사용자 요청 분석
- tasks: 작업 계획 리스트
  - task_id: 고유 ID (예: "task_1")
  - agent_name: Worker Agent 이름
  - description: 작업 설명 (구체적이고 상세하게)
  - query: 작업에 필요한 주요 쿼리 (Optional)
"""),
            ("user", "{user_request}")
        ])
        
        self.planning_chain = planning_prompt | structured_llm
        
        logger.info(f"[{self.name}] Planning Chain 구성 완료 (Function Calling)")
    
    def _get_worker_info_text(self) -> str:
        """Worker Agent 정보를 텍스트로 변환"""
        info_lines = []
        for worker in self.worker_agents.values():
            tools = list(worker.tools.keys())
            info_lines.append(f"- {worker.name}: {worker.description} (Tools: {', '.join(tools)})")
        return "\n".join(info_lines)
    
    async def plan(self, user_request: str) -> List:
        """
        작업 계획 수립
        
        Args:
            user_request: 사용자 요청
            
        Returns:
            작업 계획 리스트 (TaskPlan 객체들)
        """
        try:
            logger.info(f"[{self.name}] 작업 계획 수립 시작")
            
            # Planning Chain 실행 (ExecutionPlan 반환)
            execution_plan = await self.planning_chain.ainvoke({
                "worker_info": self._get_worker_info_text(),
                "user_request": user_request
            })
            
            logger.info(f"[{self.name}] 계획 수립 완료: {len(execution_plan.tasks)}개 작업")
            logger.info(f"[{self.name}] 분석: {execution_plan.analysis}")
            
            return execution_plan.tasks  # Pydantic 객체의 속성 접근
            
        except Exception as e:
            logger.error(f"[{self.name}] 계획 수립 실패: {str(e)}")
            raise
    
    async def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        요청 처리: plan → worker 실행 → 결과 통합
        
        Args:
            message: 사용자 요청
            context: 컨텍스트
            
        Returns:
            최종 결과
        """
        try:
            logger.info(f"[{self.name}] 요청 처리 시작: {message[:50]}...")
            
            # 1. Plan 수립
            tasks = await self.plan(message)
            
            # 2. Worker 실행 
            results = []
            previous_results = []  
            
            for idx, task in enumerate(tasks, 1):
                agent_name = task.agent_name
                worker = self.worker_agents.get(agent_name)
                
                if not worker:
                    logger.warning(f"[{self.name}] Worker 없음: {agent_name}")
                    continue
                
                logger.info(f"[{self.name}] Worker 실행 ({idx}/{len(tasks)}): {agent_name}")
                
                # Worker에게 작업 전달 
                task_message = task.description
                task_context = {
                    'task_id': task.task_id,
                    'query': task.query,
                    'previous_results': previous_results, 
                    **(context or {})
                }
                
                worker_result = await worker.process(task_message, task_context)
                
                result_entry = {
                    'task_id': task.task_id,
                    'agent_name': agent_name,
                    'result': worker_result
                }
                results.append(result_entry)
                
                if worker_result.success and worker_result.data:
                    previous_results.append({
                        'task_id': task.task_id,
                        'agent': agent_name,
                        'data': worker_result.data
                    })
            
            # 3. 결과 통합
            final_result = self._aggregate_results(results)
            
            logger.info(f"[{self.name}] 요청 처리 완료")
            
            return AgentResult(
                success=True,
                data=final_result,
                error=None
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] 요청 처리 실패: {str(e)}")
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Worker 실행 결과 통합
        
        Args:
            results: Worker 실행 결과 리스트
            
        Returns:
            통합된 결과
        """
        # 모든 Worker 결과 수집
        all_answers = []
        all_details = []
        
        for result in results:
            agent_result = result['result']
            all_details.append({
                'task_id': result['task_id'],
                'agent': result['agent_name'],
                'success': agent_result.success,
                'data': agent_result.data,
                'error': agent_result.error
            })
            
            if agent_result.success and agent_result.data:
                if isinstance(agent_result.data, dict):
                    answer = agent_result.data.get('answer')
                    if answer:
                        all_answers.append(f"[{result['agent_name']}]\n{answer}")
        
        # 최종 답변 구성
        final_answer = "\n\n".join(all_answers) if all_answers else "답변을 생성할 수 없습니다."
        
        aggregated = {
            'answer': final_answer, 
            'total_tasks': len(results),
            'successful_tasks': sum(1 for r in results if r['result'].success),
            'details': all_details  
        }
        
        return aggregated

