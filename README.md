# 차량 번호판 OCR API 시스템

## 📋 프로젝트 개요

이 프로젝트는 차량 이미지에서 한국 자동차 번호판을 자동으로 인식하는 OCR API 서비스입니다. EasyOCR과 Google Gemini API를 활용하여 2단계 인식 시스템으로 높은 정확도를 제공합니다.

## 🏗️ 시스템 아키텍처

### 핵심 구성 요소
- **FastAPI**: REST API 서버 프레임워크
- **EasyOCR**: 1단계 빠른 번호판 인식 (한글 + 영문)
- **Google Gemini API**: 2단계 정밀 번호판 인식 (폴백)
- **OpenCV**: 이미지 전처리 및 ROI 탐지
- **Docker**: 컨테이너화된 배포 환경

### 파일 구조
```
1_CAR_NO_OCR_AWS1/
├── main.py                    # FastAPI 메인 서버 파일
├── ocr_core_combine_3.py     # OCR 핵심 로직 (EasyOCR + Gemini)
├── index.html                # 웹 인터페이스
├── requirements.txt          # Python 의존성 패키지
├── Dockerfile               # Docker 컨테이너 설정
├── dockerignore            # Docker 빌드 시 제외 파일
├── aws_connect.txt         # AWS EC2 배포 가이드
└── README.md               # 이 파일
```

## 🚀 설치 및 실행 방법

### 1. 로컬 개발 환경 실행

#### 필수 요구사항
- Python 3.11 이상
- Google Gemini API 키

#### 설치 단계
```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. Google Gemini API 키 설정
# ocr_core_combine_3.py 파일의 279번째 줄에서 API 키 수정
api_key = "YOUR_GEMINI_API_KEY"

# 3. 로컬 테스트용 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 접속 방법
- 웹 인터페이스: http://16.184.13.168/
- API 문서: http://16.184.13.168/docs
- 헬스 체크: http://16.184.13.168/healthz

### 2. Docker 컨테이너 실행

#### Docker 이미지 빌드
```bash
# Docker 이미지 빌드
docker build -t lp-ocr .

# 캐시 없이 재빌드 (의존성 변경 시)
docker build --no-cache -t lp-ocr .
```

#### 컨테이너 실행 및 관리
```bash
# 컨테이너 실행 (백그라운드, 자동 재시작)
docker run -d --name lp-ocr --restart=always -p 80:8000 lp-ocr

# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 로그 확인
docker logs -f lp-ocr

# 컨테이너 중지 및 삭제
docker rm -f lp-ocr

# 모든 Docker 리소스 정리
docker system prune -a --volumes
```

### 3. AWS EC2 배포

#### 서버 접속 및 업데이트
```bash
# EC2 서버 접속
ssh -i "ec2_car_ocr.pem" ubuntu@16.184.13.168

# root 권한으로 변경
sudo su

# 프로젝트 디렉토리로 이동
cd ~/1_CAR_NO_OCR_AWS1

# Git 업데이트
git pull origin main

# Docker 재배포
docker rm -f lp-ocr 2>/dev/null || true
docker build -t lp-ocr .
docker run -d --name lp-ocr --restart=always -p 80:8000 lp-ocr
```

## 🔧 API 사용법

### 주요 엔드포인트

#### 1. 헬스 체크
```http
GET /healthz
```
**응답:** `ok`

#### 2. OCR 모델 상태 확인
```http
GET /status
```
**응답 예시:**
```json
{
  "status": "ready",
  "detail": null
}
```

#### 3. 번호판 인식
```http
POST /ocr/license-plate
Content-Type: multipart/form-data

image_file: [이미지 파일]
dryRun: false (선택사항)
```

**성공 응답 예시:**
```json
{
  "success": true,
  "plate_number": "12가3456",
  "confidence": 0.85,
  "stage": "1단계 로직",
  "elapsed_sec": 2.34
}
```

**실패 응답 예시:**
```json
{
  "success": false,
  "plate_number": null,
  "confidence": null,
  "stage": "모두 실패",
  "elapsed_sec": 5.67
}
```

### 웹 인터페이스 사용법
1. 브라우저에서 http://16.184.13.168/ 접속
2. "이미지 선택" 버튼 클릭하거나 이미지를 드래그 앤 드롭
3. 자동으로 OCR 처리 후 결과 확인
4. "워밍업" 버튼으로 모델 수동 초기화 가능

## 🧠 OCR 처리 로직

### 2단계 인식 시스템

#### 1단계: 빠른 인식 (EasyOCR)
- **엔진**: EasyOCR (한글 + 영문)
- **처리 과정**:
  1. 이미지 전처리 (그레이스케일, 블러, 이진화)
  2. ROI(관심 영역) 탐지 (윤곽선 기반)
  3. 번호판 후보 영역 추출
  4. OCR 텍스트 인식 및 정규화
  5. 한국 번호판 패턴 검증
- **성공 조건**: 신뢰도 0.6 이상
- **처리 시간**: 평균 1-3초

#### 2단계: 정밀 인식 (Gemini API)
- **엔진**: Google Gemini 2.5 Flash
- **실행 조건**: 1단계 실패 시 자동 실행
- **처리 과정**:
  1. 이미지 크기 최적화 (최대 2000px)
  2. Gemini API 호출
  3. JSON 응답 파싱
  4. 번호판 패턴 검증
- **처리 시간**: 평균 2-4초

### 번호판 인식 규칙

#### 지원 번호판 형식
- **자가용**: 가~마, 거~저, 고~조, 구~주
- **영업용**: 바, 사, 아, 자
- **택배용**: 배
- **렌터카용**: 하, 허, 호
- **특수용도**: 육, 해, 공, 국, 합

#### 정규식 패턴
```regex
^\d{2,3}[가-힣]\s?\d{4}$
```
- 2-3자리 숫자 + 한글 + 4자리 숫자
- 띄어쓰기 허용

#### 문자 보정 규칙
- **영문 → 숫자**: O→0, I→1, L→1, S→5, B→8
- **공백 제거**: 모든 공백 문자 제거
- **특수문자 제거**: 숫자와 한글만 유지

## ⚙️ 환경 설정

### 환경 변수
```bash
LOG_LEVEL=INFO                    # 로그 레벨
CORS_ORIGINS=*                    # CORS 허용 오리진
USE_GPU=false                     # GPU 사용 여부
TIMEOUT_SECONDS=90               # OCR 처리 타임아웃
```

### Google Gemini API 설정
1. Google AI Studio에서 API 키 생성
2. `ocr_core_combine_3.py` 파일의 279번째 줄에서 API 키 설정
3. Gemini 2.5 Flash 모델 사용

### EasyOCR 설정
- **언어**: 한국어(ko) + 영어(en)
- **GPU**: 기본적으로 CPU 사용 (USE_GPU=false)
- **초기화**: 서버 시작 시 백그라운드에서 자동 워밍업

## 🔍 문제 해결

### 자주 발생하는 문제

#### 1. EasyOCR 모델 로딩 실패
```
Failed to warm up EasyOCR model
```
**해결방법:**
- 인터넷 연결 확인 (모델 다운로드 필요)
- 메모리 부족 확인
- 워밍업 버튼으로 수동 재시도

#### 2. Gemini API 오류
```
Gemini API 호출 중 오류 발생
```
**해결방법:**
- API 키 유효성 확인
- API 할당량 확인
- 네트워크 연결 상태 확인

#### 3. 컨테이너 실행 오류
**해결방법:**
- 포트 80이 이미 사용 중인지 확인
- Docker 데몬이 실행 중인지 확인
- 컨테이너 로그 확인: `docker logs -f lp-ocr`

#### 4. 번호판 인식 실패
**해결방법:**
- 이미지 품질 확인 (해상도, 조명, 각도)
- 번호판이 명확히 보이는지 확인
- 지원되는 번호판 형식인지 확인

### 로그 확인 방법
```bash
# 로컬 실행 시
# 터미널에서 직접 확인

# Docker 실행 시
docker logs -f lp-ocr

# 실시간 로그 모니터링
docker logs -f --tail=100 lp-ocr
```

## 📊 성능 및 제한사항

### 성능 지표
- **1단계 처리 시간**: 평균 1-3초
- **2단계 처리 시간**: 평균 2-4초
- **전체 처리 시간**: 평균 2-5초
- **파일 크기 제한**: 10MB
- **지원 형식**: JPG, JPEG, PNG
- **타임아웃**: 90초

### 정확도
- **1단계 인식률**: 약 80-85%
- **2단계 인식률**: 약 90-95%
- **전체 인식률**: 약 85-90%
- **신뢰도 임계값**: 0.6 이상

### 제한사항
- **이미지 품질**: 저화질, 흐릿한 이미지는 인식률 저하
- **번호판 각도**: 심하게 기울어진 번호판은 인식 어려움
- **조명 조건**: 너무 어둡거나 밝은 환경은 인식률 저하
- **번호판 손상**: 가려지거나 손상된 번호판은 인식 불가
- **지원 언어**: 한국 번호판만 지원

## 🔄 업데이트 및 유지보수

### 모델 업데이트
1. EasyOCR 모델은 자동 업데이트
2. Gemini API 모델은 Google에서 자동 관리
3. 새로운 번호판 형식 추가 시 정규식 패턴 수정

### 코드 수정 시
1. 로컬에서 테스트 완료
2. Git 커밋 및 푸시
3. AWS EC2에서 `git pull` 실행
4. Docker 이미지 재빌드 및 재배포

### 모니터링
- 정기적으로 컨테이너 로그 확인
- API 응답 시간 모니터링
- 인식률 통계 수집
- 오류 발생 시 즉시 대응

## 📞 연락처 및 지원

### 기술 지원
- 시스템 관련 문의: 아우라웍스 이중재 대표
- API 사용 문의: 아우라웍스 이중재 대표

### 배포 정보
- **AWS EC2 인스턴스**: 16.184.13.168
- **SSH 키**: ec2_car_ocr_customer.pem
- **서비스 포트**: 80 (HTTP)
- **컨테이너 이름**: lp-ocr

### 문서 버전
- 작성일: 2025년 9월 7일
- 최종 수정일: 2025년 9월 7일
- 버전: 1.0.0

---

**⚠️ 주의사항**
- Google Gemini API 사용량에 따라 비용이 발생할 수 있습니다.
- 프로덕션 환경에서는 API 키를 환경 변수로 관리하는 것을 권장합니다.
- 정기적인 성능 모니터링 및 인식률 평가가 필요합니다.
- AWS EC2 인스턴스 비용을 고려하여 사용량을 모니터링하세요.
