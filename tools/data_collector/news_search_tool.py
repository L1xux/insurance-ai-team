"""
News Search Tool
=========================
Author: Jin
Date: 2026.02.07
Version: 1.1 (Fixed Input Handling)

Description:
GNews 라이브러리를 활용한 뉴스 크롤링 및 시장 동향 분석 도구입니다.
보험 상품 기획을 위한 경쟁사 동향, 규제 정보, 소비자 트렌드를 수집합니다.
"""
from typing import Optional, Dict, Any, List, Set
import json
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# External dependency: gnews
from gnews import GNews
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from config.logging_config import logger
from models.tool_model import ToolSchema

class NewsSearchTool(BaseTool):
    """뉴스 검색 및 시장 동향 분석 Tool"""
    
    name: str = "news_market_search"
    description: str = "글로벌 뉴스 및 시장 동향 검색 도구. 입력은 JSON({'keywords': [...]}) 또는 단순 문자열 키워드 모두 가능합니다."
    
    def __init__(self):
        super().__init__()
        logger.info(f"[{self.name}] 초기화 완료")

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """뉴스 검색 실행 (입력 예외 처리 강화)"""
        try:
            # 기본값 설정
            keywords = []
            date_range = '7d'
            max_results = 10

            # 1. 입력값 파싱 및 타입 체크
            try:
                if query is None:
                    return json.dumps({'success': False, 'error': '검색 키워드가 없습니다 (Input is None).'})

                if isinstance(query, dict):
                    params = query
                else:
                    params = json.loads(query)

                # 파싱 결과가 딕셔너리인지 확인
                if isinstance(params, dict):
                    keywords = params.get('keywords', [])
                    date_range = params.get('date_range', '7d')
                    max_results = params.get('max_results', 10)
                    
                    if isinstance(keywords, str):
                        keywords = [keywords]
                else:
                    # JSON 파싱 결과가 문자열인 경우
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

            logger.info(f"[{self.name}] 검색 요청: {keywords}, 기간: {date_range}")

            # GNews 라이브러리용 기간 형식으로 변환
            period = self._convert_date_range_to_period(date_range)
            
            # 병렬 검색 수행
            articles = self._search_parallel(keywords, period, max_results)
            
            # 소스별 통계 생성
            source_stats = self._generate_source_stats(articles)
            
            return json.dumps({
                'success': True,
                'total_count': len(articles),
                'period_setting': period,
                'articles': articles,
                'source_stats': source_stats
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
        period: str, 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """키워드별 병렬 검색 및 데이터 정제"""
        
        all_articles = []
        seen_urls = set()
        
        def search_keyword(kw):
            google_news = GNews(
                language='en',
                country='US',
                period=period,
                max_results=max_results,
                exclude_websites=[]
            )
            results = []
            try:
                raw_data = google_news.get_news(kw)
                if raw_data:
                    results = raw_data
            except Exception as e:
                logger.warning(f"[{self.name}] '{kw}' 검색 실패: {e}")
            return results

        with ThreadPoolExecutor(max_workers=min(len(keywords), 5)) as executor:
            future_to_kw = {executor.submit(search_keyword, kw): kw for kw in keywords}
            
            for future in future_to_kw:
                try:
                    results = future.result()
                    for item in results:
                        url = item.get('url')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            
                            raw_date = item.get('published date', '')
                            normalized_date = self._normalize_date(raw_date)
                            
                            article_data = {
                                "title": item.get('title', ''),
                                "source": item.get('publisher', {}).get('title', 'Unknown'),
                                "published": normalized_date,
                                "url": url,
                                "description": item.get('description', '')
                            }
                            all_articles.append(article_data)
                except Exception as e:
                    kw = future_to_kw[future]
                    logger.error(f"[{self.name}] '{kw}' 처리 중 오류: {e}")

        all_articles.sort(key=lambda x: x['published'], reverse=True)
        return all_articles

    def _normalize_date(self, date_str: str) -> str:
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")
        try:
            if "," in date_str:
                dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                return dt.strftime("%Y-%m-%d")
            return date_str
        except ValueError:
            return datetime.now().strftime("%Y-%m-%d")

    def _convert_date_range_to_period(self, date_range: str) -> str:
        date_range = str(date_range).lower()
        if re.match(r'^\d+[dmy]$', date_range): return date_range
        if "year" in date_range:
            num = re.search(r'\d+', date_range)
            return f"{num.group() if num else '1'}y"
        if "month" in date_range:
            num = re.search(r'\d+', date_range)
            return f"{num.group() if num else '1'}m"
        if "day" in date_range:
            num = re.search(r'\d+', date_range)
            return f"{num.group() if num else '7'}d"
        return "7d"

    def _generate_source_stats(self, articles: List[Dict]) -> Dict[str, int]:
        stats = {}
        for a in articles:
            src = a.get('source', 'Unknown')
            stats[src] = stats.get(src, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="뉴스 검색 - 글로벌 시장 동향, 경쟁사 뉴스",
            parameters={
                'keywords': {'type': 'array', 'items': {'type': 'string'}},
                'date_range': {'type': 'string'},
                'max_results': {'type': 'integer'}
            },
            required_params=['keywords']
        )