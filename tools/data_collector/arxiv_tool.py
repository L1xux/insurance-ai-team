"""
Arxiv Search Tool
=========================
Author: Jin
Date: 2026.02.07
Version: 1.1 (Fixed Input Handling)

Description:
보험 상품 기획 및 리스크 분석을 위한 arXiv 논문 검색 도구입니다.
사용자 요청(키워드, 날짜 등)을 받아 병렬로 논문을 검색하고 메타데이터를 추출합니다.
"""
from typing import Optional, Dict, Any, List, Set
import json
import re
import arxiv
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from config.logging_config import logger
from models.tool_model import ToolSchema

class ArxivSearchTool(BaseTool):
    """arXiv 논문 검색 및 메타데이터 추출 Tool"""
    
    name: str = "arxiv_paper_search"
    description: str = "arXiv 논문 검색 도구. 입력은 JSON({'keywords': [...]}) 또는 단순 문자열 키워드 모두 가능합니다."
    
    client: Any = None
    company_patterns: List[Any] = []
    
    TARGET_COMPANIES: Set[str] = {
        "Allianz", "AXA", "Prudential", "MetLife", "Cigna", "UnitedHealth",
        "Munich Re", "Swiss Re", "Ping An", "China Life", "AIA",
        "Zurich Insurance", "Chubb", "Manulife", "Generali",
        "Lemonade", "Root Insurance", "Oscar Health",
        "Google", "Amazon", "Microsoft", "Tesla", "IBM", "Apple"
    }

    def __init__(self):
        """Tool 초기화"""
        super().__init__()
        self.client = arxiv.Client(
            page_size=50,
            delay_seconds=3,
            num_retries=3
        )
        self._compile_patterns()
        logger.info(f"[{self.name}] 초기화 완료")

    def _compile_patterns(self) -> None:
        """기업명 매칭을 위한 정규식 컴파일"""
        self.company_patterns = [
            re.compile(r'\b' + re.escape(company) + r'\b', re.IGNORECASE)
            for company in self.TARGET_COMPANIES
        ]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """논문 검색 실행 (입력 예외 처리 강화)"""
        try:
            # 기본값 설정
            keywords = []
            date_range = '3 years'
            max_results = 50

            # 1. 입력값 파싱 및 타입 체크
            try:
                # 입력이 None인 경우
                if query is None:
                    return json.dumps({'success': False, 'error': '검색 키워드가 없습니다 (Input is None).'})

                # 입력이 이미 딕셔너리인 경우 (일부 Agent 실행 환경 대응)
                if isinstance(query, dict):
                    params = query
                else:
                    # JSON 파싱 시도
                    params = json.loads(query)

                # 파싱 결과가 딕셔너리인지 확인
                if isinstance(params, dict):
                    keywords = params.get('keywords', [])
                    date_range = params.get('date_range', '3 years')
                    max_results = params.get('max_results', 50)
                    
                    # keywords가 문자열 하나로 온 경우 리스트로 변환
                    if isinstance(keywords, str):
                        keywords = [keywords]
                else:
                    # JSON 파싱은 됐는데 딕셔너리가 아님 (예: "search query" -> 문자열)
                    keywords = [str(params)]

            except (json.JSONDecodeError, TypeError):
                # JSON 파싱 실패 -> 입력값을 그대로 키워드로 사용
                keywords = [str(query)]

            # 2. 키워드 유효성 검사
            if not keywords:
                return json.dumps({
                    'success': False,
                    'error': '검색 키워드가 제공되지 않았습니다.'
                }, ensure_ascii=False)

            logger.info(f"[{self.name}] 검색 요청: {keywords}, 범위: {date_range}")

            # 3. 검색 실행
            start_date, end_date = self._parse_date_range(date_range)
            search_results = self._search_parallel(keywords, start_date, end_date, max_results)
            
            # 결과 요약 통계
            total_papers = len(search_results)
            company_stats = self._generate_company_stats(search_results)
            
            return json.dumps({
                'success': True,
                'total_count': total_papers,
                'date_range': f"{start_date} to {end_date}",
                'papers': search_results,
                'company_stats': company_stats
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"[{self.name}] 실행 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)

    def _search_parallel(
        self, 
        keywords: List[str], 
        start_date: datetime.date, 
        end_date: datetime.date,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """키워드 병렬 검색 및 데이터 정제"""
        
        all_papers = []
        seen_ids = set()
        
        def search_keyword(kw):
            query = f'all:"{kw}"'
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = []
            try:
                for result in self.client.results(search):
                    published = result.published.date()
                    if not (start_date <= published <= end_date):
                        continue
                    results.append(result)
            except Exception as e:
                logger.warning(f"[{self.name}] '{kw}' 검색 중 부분 오류: {e}")
                
            return results

        with ThreadPoolExecutor(max_workers=min(len(keywords), 5)) as executor:
            future_to_kw = {executor.submit(search_keyword, kw): kw for kw in keywords}
            
            for future in future_to_kw:
                try:
                    results = future.result()
                    for res in results:
                        if res.entry_id not in seen_ids:
                            seen_ids.add(res.entry_id)
                            
                            text_content = f"{res.title} {res.summary}"
                            companies = self._extract_companies(text_content)
                            tech_keywords = self._extract_tech_keywords(res.title, res.summary)
                            
                            paper_data = {
                                "title": res.title,
                                "authors": [a.name for a in res.authors],
                                "published": res.published.strftime("%Y-%m-%d"),
                                "summary": res.summary.replace("\n", " "),
                                "url": res.entry_id,
                                "companies": companies,
                                "keywords": tech_keywords
                            }
                            all_papers.append(paper_data)
                            
                except Exception as e:
                    kw = future_to_kw[future]
                    logger.error(f"[{self.name}] '{kw}' 처리 실패: {e}")

        all_papers.sort(key=lambda x: x['published'], reverse=True)
        return all_papers

    def _extract_companies(self, text: str) -> List[str]:
        found = set()
        text_lower = text.lower()
        for company in self.TARGET_COMPANIES:
            if company.lower() in text_lower:
                found.add(company)
        return list(found)

    def _extract_tech_keywords(self, title: str, summary: str) -> List[str]:
        text = f"{title} {summary}"
        pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b'
        matches = re.findall(pattern, text)
        stopwords = {"The", "This", "In", "On", "For", "To", "From", "We", "Our"}
        filtered = {m for m in matches if m.split()[0] not in stopwords}
        return list(filtered)[:5]

    def _parse_date_range(self, date_range: str) -> tuple:
        try:
            today = datetime.now().date()
            if "year" in date_range:
                years = int(re.search(r'\d+', date_range).group())
                start_date = today - timedelta(days=years * 365)
                return start_date, today
            if "to" in date_range:
                start_str, end_str = date_range.split(" to ")
                start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
                end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
                return start_date, end_date
            return today - timedelta(days=365 * 3), today
        except Exception as e:
            logger.warning(f"[{self.name}] 날짜 파싱 실패 ('{date_range}'), 기본값 사용: {e}")
            return datetime.now().date() - timedelta(days=365 * 3), datetime.now().date()

    def _generate_company_stats(self, papers: List[Dict]) -> Dict[str, int]:
        stats = {}
        for p in papers:
            for c in p['companies']:
                stats[c] = stats.get(c, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="arXiv 논문 검색 - 키워드 기반 학술 자료 검색",
            parameters={
                'keywords': {'type': 'array', 'items': {'type': 'string'}},
                'date_range': {'type': 'string'},
                'max_results': {'type': 'integer'}
            },
            required_params=['keywords']
        )