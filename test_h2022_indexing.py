"""
H2022 인덱싱 테스트 스크립트
=========================
Author: Jin
Date: 2025.10.12

Description:
H2022 문서 인덱싱 파이프라인을 테스트합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.insurance.h2022.h2022_embedding import H2022Embedding
from rag.insurance.h2022.h2022_vector_store import H2022VectorStore
from rag.insurance.h2022.h2022_indexer import H2022Indexer
from config.logging_config import logger


def main():
    """메인 함수"""
    
    print("=" * 80)
    print("H2022 문서 인덱싱 테스트 시작")
    print("=" * 80)
    
    try:
        # 1. H2022 임베딩 모델 초기화
        print("\n[1단계] 임베딩 모델 초기화 중...")
        embedder = H2022Embedding(
            model_name="text-embedding-3-small",
            dimension=1536
        )
        print(f"✓ 임베딩 모델 초기화 완료 (차원: {embedder.dimension})")
        
        # 2. H2022 벡터 저장소 초기화
        print("\n[2단계] 벡터 저장소 초기화 중...")
        vector_store = H2022VectorStore(
            dimension=1536,
            table_name="h2022_vector_embeddings"
        )
        print(f"✓ 벡터 저장소 초기화 완료 (현재 벡터 수: {vector_store.vector_count})")
        
        # 3. H2022 인덱서 초기화
        print("\n[3단계] 인덱서 초기화 중...")
        indexer = H2022Indexer(
            embedder=embedder,
            vector_store=vector_store,
            chunk_size=1000,
            chunk_overlap=200
        )
        print(f"✓ 인덱서 초기화 완료")
        
        # 4. PDF 파일 경로 확인
        pdf_path = project_root / "data" / "h2022doc.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        print(f"\n[4단계] PDF 파일 확인: {pdf_path}")
        print(f"✓ 파일 존재 확인 완료 (크기: {pdf_path.stat().st_size:,} bytes)")
        
        # 5. 문서 인덱싱 시작
        print("\n[5단계] 문서 인덱싱 시작...")
        print("-" * 80)
        
        result = indexer.index_document(
            filepath=str(pdf_path),
            metadata={
                "source": "MEPS_HC243_2022",
                "description": "Medical Expenditure Panel Survey - 2022 Full Year Consolidated Data File"
            }
        )
        
        print("-" * 80)
        
        # 6. 결과 출력
        print("\n[6단계] 인덱싱 결과")
        print("=" * 80)
        
        if result.get('success'):
            print("✓ 인덱싱 성공!")
            print(f"\n📊 통계:")
            print(f"  - 파일: {result.get('filepath')}")
            print(f"  - 처리된 문서(페이지) 수: {result.get('documents_count', 0)}")
            print(f"  - 생성된 청크 수: {result.get('chunks_count', 0)}")
            print(f"  - 벡터 저장소 총 벡터 수: {result.get('vector_store_total', 0)}")
            
            # 샘플 ID 출력
            doc_ids = result.get('doc_ids', [])
            if doc_ids:
                print(f"\n📝 생성된 청크 ID 샘플 (처음 5개):")
                for i, doc_id in enumerate(doc_ids[:5], 1):
                    print(f"  {i}. {doc_id}")
                
                if len(doc_ids) > 5:
                    print(f"  ... (나머지 {len(doc_ids) - 5}개)")
            
            print("\n" + "=" * 80)
            print("✅ 테스트 완료!")
            print("=" * 80)
            
        else:
            print("❌ 인덱싱 실패")
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
            return 1
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일 오류: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        logger.exception("인덱싱 테스트 중 오류 발생")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

