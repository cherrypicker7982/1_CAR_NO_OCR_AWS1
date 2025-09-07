import os
import time
import logging
import asyncio
import mimetypes
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread

# easyocr 임포트는 파일 최상단에서 한 번만 합니다.
import easyocr

# --- 환경 변수에서 설정 읽기 ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(',')
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 90))

# --- 로깅 설정 ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s] [%(process)d] [%(threadName)s] [%(name)s] %(message)s",
)
log = logging.getLogger("uvicorn")

app = FastAPI(
    title="OCR API",
    description="A FastAPI service for license plate OCR.",
    version="1.0.0"
)

# --- CORS 미들웨어 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 헬스 체크 및 기본 경로 ---
@app.get("/healthz", response_class=PlainTextResponse, summary="Health check endpoint")
def healthz():
    """Simple health check for load balancers."""
    return "ok"

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(Path(__file__).parent / "index.html")

# --- EasyOCR 워밍업 및 상태 관리 ---
OCR_READY = False
OCR_LOADING = False
OCR_ERROR = None
reader = None  # EasyOCR reader 객체를 저장할 전역 변수 추가

def _warmup_easyocr():
    """Background task to load the EasyOCR model."""
    global OCR_READY, OCR_LOADING, OCR_ERROR, reader
    
    if OCR_READY or OCR_LOADING:
        log.info("OCR model is already being loaded or is ready.")
        return
    
    OCR_LOADING = True
    OCR_ERROR = None
    start_time = time.time()
    log.info("Starting EasyOCR model warmup...")

    try:
        reader = easyocr.Reader(['en', 'ko'], gpu=USE_GPU, verbose=False)
        
        # 실제 워밍업을 위해 간단한 이미지 파일을 읽어볼 수 있지만,
        # 파일이 없을 경우 오류가 발생하므로 주석 처리하는 것이 더 안전합니다.
        # reader.readtext('dummy.png', detail=0)

        log.info("EasyOCR reader initialized successfully.")
        
        OCR_READY = True
        elapsed_time = time.time() - start_time
        log.info(f"EasyOCR model is ready. Warmup took {elapsed_time:.2f} seconds.")
    except Exception as e:
        OCR_ERROR = repr(e)
        log.exception("Failed to warm up EasyOCR model.")
    finally:
        OCR_LOADING = False

@app.on_event("startup")
def start_warmup_thread():
    """Start the warmup process in a background thread."""
    thread = Thread(target=_warmup_easyocr, daemon=True, name="EasyOCR-Warmup")
    thread.start()

@app.get("/status", summary="Get OCR model status")
def get_status():
    """Provides the current status of the OCR model."""
    return {
        "status": "ready" if OCR_READY else "loading" if OCR_LOADING else "error",
        "detail": OCR_ERROR,
    }

# --- OCR 엔드포인트 ---
def _get_file_suffix(filename: str, content_type: str) -> str:
    """Determine file suffix based on filename or content type."""
    if filename and "." in filename:
        return "." + filename.split(".")[-1].lower()
    return mimetypes.guess_extension(content_type) or ".bin"

# def _run_ocr_blocking(temp_path: str):
#     """Blocking OCR logic to be run in a separate thread."""
#     global reader # 전역 reader 객체를 사용하도록 선언
#     try:
#         from ocr_core_combine_3 import recognize_plate_combined
#         # recognize_plate_combined 함수에 reader 객체를 전달하도록 수정
#         return recognize_plate_combined(temp_path, debug=False, reader=reader)
#     finally:
#         # 이 부분은 변경 없음
#         os.remove(temp_path)
#         log.info(f"Temporary file removed: {temp_path}")
        
    
def _run_ocr_blocking(temp_path: str, reader):
# """Blocking OCR logic to be run in a separate thread."""
    try:
        from ocr_core_combine_3 import recognize_plate_combined
        return recognize_plate_combined(temp_path, debug=False, reader=reader)
    finally:
        os.remove(temp_path)
        log.info(f"Temporary file removed: {temp_path}")
        
        

@app.post("/ocr/license-plate", summary="Recognize license plates from an image")
async def recognize_license_plate(
    image_file: UploadFile = File(...),
    dry_run: bool = Query(False, description="If true, only check file upload without running OCR."),
):
    """
    Receives an image file and performs OCR to recognize license plates.
    
    - **dry_run**: A query parameter to test the file upload process without the OCR model.
    - **503 Service Unavailable**: Returned if the OCR model is still loading.
    - **504 Gateway Timeout**: Returned if the OCR process exceeds the timeout.
    """
    log.info(f"OCR request received: dry_run={dry_run}, filename={image_file.filename}, content_type={image_file.content_type}")

    if not OCR_READY and not dry_run:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR model is still loading. Please try again in a moment."
        )

    # 1. 파일 처리
    try:
        data = await image_file.read()
    except Exception as e:
        log.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read file.")

    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is not an image.")

    # 2. 임시 파일 저장
    suffix = _get_file_suffix(image_file.filename, image_file.content_type)
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name
        log.info(f"Temporary file saved: {temp_path} ({len(data)} bytes)")
    except Exception as e:
        log.error(f"Failed to save temporary file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save file on server.")
    
    # 2.5. 드라이런
    if dry_run:
        os.remove(temp_path)
        return {"message": "Dry run successful.", "file_size": len(data)}

    # 3. OCR 실행
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            # loop.run_in_executor(None, _run_ocr_blocking, temp_path),
            loop.run_in_executor(None, _run_ocr_blocking, temp_path, reader),
            timeout=TIMEOUT_SECONDS
        )
        return result
    except ImportError:
        log.error("OCR module 'ocr_core_combine_3' not found.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OCR module not found.")
    except asyncio.TimeoutError:
        log.error(f"OCR process timed out after {TIMEOUT_SECONDS} seconds.")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="OCR processing timed out.")
    except Exception as e:
        log.error(f"An error occurred during OCR: {e}")
        # 오류 메시지 자체를 반환하도록 수정
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during OCR processing: {e}")