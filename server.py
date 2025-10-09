"""
FastAPI 기반 데이터 분석 실행 서버
=========================
Author: Jin
Date: 2025.10.07
Version: 2.0

Description:
기존 main.py의 로직을 FastAPI로 래핑하여 웹에서 실행 가능하게 함
"""
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
import os
import asyncio
import json
from typing import Optional

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 기존 main.py에서 사용하던 모듈들 임포트
from utils.handler.csv_handler import CsvHandler
from utils.handler.llm_model_handler import LLMModelHandler
from factories.service_factory import ServiceFactory
from services.llm.llm_service import LLMService
from config.logging_config import logger

# 전역 변수
available_files = []
available_models = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 lifespan 이벤트 핸들러"""
    global available_files, available_models
    
    # Startup
    logger.info("서버 시작 - 파일 및 모델 목록 로드 중...")
    
    try:
        # CSV 파일 목록 가져오기
        csv_handler = CsvHandler(folder_path=current_dir)
        csv_handler.save_all_csv()
        available_files = csv_handler.csv_files
        
        # LLM 모델 목록 가져오기
        llm_handler = LLMModelHandler()
        available_models = llm_handler.models
        
        logger.info(f"CSV 파일 {len(available_files)}개 로드")
        logger.info(f"사용 가능한 모델 타입: {type(available_models)}")
        if available_models:
            logger.info(f"첫 번째 모델 샘플: {available_models[0] if available_models else 'None'}")
        
    except Exception as e:
        logger.error(f"초기화 오류: {str(e)}", exc_info=True)
        available_files = []
        available_models = []
    
    yield  # 여기서 서버 실행
    
    # Shutdown (필요시)
    logger.info("서버 종료")


# FastAPI 앱 생성 (lifespan 핸들러 포함)
app = FastAPI(
    title="Data Analysis Runner API",
    description="기존 데이터 분석 로직을 웹으로 실행",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 시각화 이미지 디렉토리 설정 및 정적 파일 서빙
VISUALIZATION_DIR = os.path.join(current_dir, "data", "visualizations")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=VISUALIZATION_DIR), name="images")

# 저장 디렉토리 생성
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>데이터 분석 실행 해요</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, button {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .loading {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .model-info {
            font-size: 12px;
            color: #777;
            margin-top: 3px;
        }
        #visualizationContainer {
            margin-top: 30px;
            display: none;
        }
        #visualizationImage {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .image-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>데이터 분석</h1>
        
        <form id="analysisForm">
            <div class="form-group">
                <label for="csv_file">CSV 파일 선택:</label>
                <select id="csv_file" name="csv_file" required>
                    <option value="">-- 파일을 선택하세요 --</option>
                    {{CSV_FILES}}
                </select>
            </div>
            
            <div class="form-group">
                <label for="llm_model">LLM 모델:</label>
                <select id="llm_model" name="llm_model" required>
                    <option value="">-- 모델을 선택하세요 --</option>
                    {{LLM_MODELS}}
                </select>
            </div>
            
            <button type="submit" id="runButton">분석 실행</button>
        </form>
        
        <div id="result"></div>
        
        <!-- 시각화 이미지 표시 영역 -->
        <div id="visualizationContainer">
            <div class="image-title">생성된 시각화</div>
            <img id="visualizationImage" alt="데이터 시각화 결과" />
        </div>
    </div>

    <script>
        const form = document.getElementById('analysisForm');
        const resultDiv = document.getElementById('result');
        const runButton = document.getElementById('runButton');
        const vizContainer = document.getElementById('visualizationContainer');
        const vizImage = document.getElementById('visualizationImage');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(form);
            const csv_file = formData.get('csv_file');
            const llm_model = formData.get('llm_model');
            
            if (!csv_file || !llm_model) {
                showResult('모든 필드를 선택해주세요.', 'error');
                return;
            }
            
            // 이전 결과 초기화
            vizContainer.style.display = 'none';
            runButton.disabled = true;
            showResult('<div class="spinner"></div> 분석을 실행 중입니다... 잠시만 기다려주세요.', 'loading');
            
            try {
                const response = await fetch('/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        csv_file: csv_file,
                        llm_model: llm_model
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // 성공 메시지 표시
                    showResult(
                        data.message + '<br><br>' + 
                        '<strong>파일:</strong> ' + data.details + '<br>' +
                        '<strong>모델:</strong> ' + data.provider.toUpperCase() + ' - ' + data.model, 
                        'success'
                    );
                    
                    // 이미지 경로에서 파일명 추출
                    // 예: data/visualizations/visualization_20251007_153949.png -> visualization_20251007_153949.png
                    const filePath = data.file_path;
                    const fileName = filePath.split('/').pop();
                    
                    console.log('파일 경로:', filePath);
                    console.log('파일명:', fileName);
                    
                    // 이미지 URL 생성 및 표시
                    const imageUrl = `/images/${fileName}`;
                    console.log('이미지 URL:', imageUrl);
                    
                    vizImage.src = imageUrl;
                    vizContainer.style.display = 'block';
                    
                    // 이미지 로딩 에러 처리
                    vizImage.onerror = function() {
                        console.error('이미지 로딩 실패:', imageUrl);
                        showResult('분석은 완료되었으나 이미지를 표시할 수 없습니다.<br>' + data.details, 'error');
                        vizContainer.style.display = 'none';
                    };
                    
                    // 이미지 로딩 성공
                    vizImage.onload = function() {
                        console.log('이미지 로딩 성공!');
                    };
                    
                } else {
                    showResult('오류: ' + data.detail, 'error');
                }
            } catch (error) {
                showResult('오류: ' + error.message, 'error');
            } finally {
                runButton.disabled = false;
            }
        });
        
        function showResult(message, type) {
            resultDiv.innerHTML = message;
            resultDiv.className = type;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지 - 파일 선택 및 실행 UI"""
    
    # CSV 파일 옵션 생성
    csv_options = ""
    for file_path in available_files:
        filename = os.path.basename(file_path)
        csv_options += f'<option value="{file_path}">{filename}</option>\n'
    
    # LLM 모델 옵션 생성 (available_models는 단순 문자열 리스트)
    model_options = ""
    
    try:
        logger.info(f"available_models 타입: {type(available_models)}")
        logger.info(f"available_models: {available_models}")
        
        if isinstance(available_models, list) and available_models:
            # LLMModelHandler의 models는 문자열 리스트
            for model in available_models:
                # MODEL_TO_PROVIDER에서 provider 찾기
                provider = LLMModelHandler.MODEL_TO_PROVIDER.get(model, "unknown")
                description = LLMModelHandler.MODEL_DESCRIPTIONS.get(model, model)
                
                model_options += f'<option value="{model}">{description} ({provider.upper()})</option>\n'
        else:
            logger.warning(f"예상치 못한 models 형식: {type(available_models)}")
            
    except Exception as e:
        logger.error(f"모델 목록 생성 오류: {str(e)}", exc_info=True)
    
    html = HTML_TEMPLATE.replace("{{CSV_FILES}}", csv_options)
    html = html.replace("{{LLM_MODELS}}", model_options)
    
    return html


@app.post("/run")
async def run_analysis(
    csv_file: str = Form(...),
    llm_model: str = Form(...)
):
    """
    기존 main.py의 메인 로직 실행
    """
    logger.info("데이터 분석 프로젝트 시작")
    
    try:
        # 1. CSV 파일 검증
        if not os.path.exists(csv_file):
            raise HTTPException(status_code=404, detail="선택한 파일을 찾을 수 없습니다.")
        
        logger.info(f"선택된 파일: {csv_file}")
        
        # 2. 모델에서 Provider 자동 결정
        provider = LLMModelHandler.MODEL_TO_PROVIDER.get(llm_model)
        if not provider:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 모델입니다: {llm_model}"
            )
        
        logger.info(f"선택된 모델: {provider} - {llm_model}")
        
        # 3. 서비스 생성 및 분석 (기존 main.py 로직)
        factory = ServiceFactory()
        service_type = factory.detect_service_type(csv_file)
        service = factory.create_service(service_type)
        
        logger.info("데이터 분석 중...")
        results = service.execute(csv_file)
        
        # 4. LLM 시각화 (기존 main.py 로직)
        logger.info(f"{provider.upper()} {llm_model}로 시각화 생성 중...")
        llm_service = LLMService(provider=provider, model_name=llm_model)
        
        df = service.loader.df if service.loader else None
        if df is None:
            raise HTTPException(status_code=500, detail="데이터프레임을 로드할 수 없습니다.")
        
        viz_result = await llm_service.create_visualization_with_user_input(
            results.get('analysis', {}),
            df,
            interactive=False
        )
        
        await llm_service.close()
        
        # 5. 결과 반환
        if viz_result['success']:
            message = "분석 및 시각화가 완료되었습니다!"
            details = f"파일 저장 위치: {viz_result['file_path']}"
            logger.info(f"완료: {viz_result['file_path']}")
            
            return JSONResponse({
                "success": True,
                "message": message,
                "details": details,
                "file_path": str(viz_result['file_path']),
                "provider": str(provider),
                "model": str(llm_model)
            })
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"시각화 생성 실패: {viz_result['error_message']}"
            )
        
    except HTTPException:
        raise
    except KeyboardInterrupt:
        logger.info("프로그램이 중단되었습니다.")
        raise HTTPException(status_code=500, detail="프로그램이 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")


@app.get("/files")
async def list_files():
    """사용 가능한 CSV 파일 목록 반환"""
    files = [{"path": f, "name": os.path.basename(f)} for f in available_files]
    return JSONResponse({"files": files})


@app.get("/models")
async def list_models():
    """사용 가능한 LLM 모델 목록 반환"""
    models = []
    for model in available_models:
        provider = LLMModelHandler.MODEL_TO_PROVIDER.get(model, "unknown")
        description = LLMModelHandler.MODEL_DESCRIPTIONS.get(model, model)
        models.append({
            "model": model,
            "provider": provider,
            "description": description
        })
    return JSONResponse({"models": models})


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return JSONResponse({
        "status": "healthy",
        "available_files": len(available_files),
        "available_models": len(available_models)
    })


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)