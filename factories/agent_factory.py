"""
Agent Factory 
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
Agent와 Tool을 생성하고 의존성을 주입합니다.
"""

from typing import Dict, List, Optional

from base.agent.agent_base import ManagerAgent, WorkerAgent
from base.agent.llm_base import LLMBase
from config.logging_config import logger
from langchain_core.tools import BaseTool

class AgentFactory:
    """Agent Factory"""
    
    def __init__(self):
        """Factory 초기화"""
        self._tools: Dict[str, BaseTool] = {}
        self._llms: Dict[str, LLMBase] = {}
        self._workers: Dict[str, WorkerAgent] = {}
        self._managers: Dict[str, ManagerAgent] = {}
        logger.info("[AgentFactory] 초기화 완료")
    
    # ========== Tool ==========    
    def register_tool(self, tool: BaseTool) -> None:
        """Tool 등록"""
        self._tools[tool.name] = tool
        logger.info(f"[AgentFactory] Tool 등록: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Tool 조회"""
        return self._tools.get(name)
    
    def get_tools(self, names: List[str]) -> List[BaseTool]:
        """여러 Tool 조회"""
        tools = []
        for name in names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"[AgentFactory] Tool 없음: {name}")
        return tools
    
    def get_all_tool_names(self) -> List[str]:
        """등록된 모든 Tool 이름 반환"""
        return list(self._tools.keys())
    
    def get_all_tools(self) -> List[BaseTool]:
        """등록된 모든 Tool 반환"""
        return list(self._tools.values())
    
    # ========== LLM ==========
    def register_llm(self, name: str, llm: LLMBase) -> None:
        """LLM 등록"""
        self._llms[name] = llm
        logger.info(f"[AgentFactory] LLM 등록: {name}")
    
    def get_llm(self, name: str) -> Optional[LLMBase]:
        """LLM 조회"""
        return self._llms.get(name)
    
    # ========== Worker Agent ==========
    def create_worker(
        self,
        name: str,
        description: str,
        tool_names: List[str],
        llm_name: List[str],
        worker_class: type
    ) -> WorkerAgent:
        """
        Worker Agent 생성
        
        Args:
            name: Agent 이름
            description: Agent 설명
            tool_names: Tool 이름 리스트
            llm_name: LLM 이름
            worker_class: Worker Agent 클래스
            
        Returns:
            생성된 Worker Agent
        """
        # Tool 주입
        tools = self.get_tools(tool_names)
        
        # LLM 주입 
        llm = None
        if llm_name:
            llm = self.get_llm(llm_name)
            if not llm:
                logger.warning(f"[AgentFactory] LLM 없음: {llm_name}")
        
        # Worker 생성
        worker = worker_class(
            name=name,
            description=description,
            tools=tools,
            llm=llm
        )
        
        self._workers[name] = worker
        logger.info(f"[AgentFactory] Worker 생성: {name}")
        
        return worker
    
    def get_worker(self, name: str) -> Optional[WorkerAgent]:
        """Worker Agent 조회"""
        return self._workers.get(name)
    
    def get_workers(self, names: List[str]) -> List[WorkerAgent]:
        """여러 Worker Agent 조회"""
        workers = []
        for name in names:
            worker = self.get_worker(name)
            if worker:
                workers.append(worker)
            else:
                logger.warning(f"[AgentFactory] Worker 없음: {name}")
        return workers
    
    # ========== Manager Agent ==========  
    def create_manager(
        self,
        name: str,
        description: str,
        worker_names: List[str],
        llm_name: str,
        manager_class: type
    ) -> ManagerAgent:
        """
        Manager Agent 생성
        
        Args:
            name: Agent 이름
            description: Agent 설명
            worker_names: Worker Agent 이름 리스트
            llm_name: LLM 이름
            manager_class: Manager Agent 클래스
            
        Returns:
            생성된 Manager Agent
        """
        # Worker 주입
        workers = self.get_workers(worker_names)
        if not workers:
            raise ValueError(f"Manager에 Worker가 필요합니다: {name}")
        
        # LLM 주입
        llm = self.get_llm(llm_name)
        if not llm:
            raise ValueError(f"LLM 없음: {llm_name}")
        
        # Manager 생성
        manager = manager_class(
            name=name,
            description=description,
            worker_agents=workers,
            llm=llm
        )
        
        self._managers[name] = manager
        logger.info(f"[AgentFactory] Manager 생성: {name}")
        
        return manager
    
    def get_manager(self, name: str) -> Optional[ManagerAgent]:
        """Manager Agent 조회"""
        return self._managers.get(name)


# Singleton
_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Factory Singleton 반환"""
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory


def reset_agent_factory() -> None:
    """Factory 리셋"""
    global _factory
    _factory = None
