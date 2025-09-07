# ocr_core_combined.py
# ocr_core_combine_2.py :  2단계 없앰. 1단계만 사용.
# ocr_core_combine_3.py :  2단계 삭제. gemini 로직 추가  // 가로픽셀 2천 다운사이징 로직 추가.
# ocr_core_combined.py
# ocr_core_combine_4_1.py : Google Cloud Vision API 로직으로 변경
# ocr_core_combine_4_1.py : Google Cloud Vision API 로직으로 변경 : 문제있어서 사용하지 않음!!!
#                           사유 : 번호판 내 '한글' 인식률이 매우 떨어짐. why? 
# # 41. 기술적 접근 방식의 차이
# Google Cloud Vision API (GCP): GCV는 주로 이미지 내의 텍스트를 "정적"으로 인식하는 데 특화된 전통적인 OCR 엔진입니다. 글자 하나하나를 모양에 기반하여 인식하고, 그 결과를 단순 텍스트로 반환합니다. 이 과정에서 시각적으로 비슷하게 생긴 한글('러')과 숫자('2')를 혼동하는 오인식이 발생할 수 있습니다.
# Gemini API (제공된 코드 기준): Gemini는 단순 OCR을 넘어선 "멀티모달 대규모 언어 모델(LLM)"입니다. 이미지와 텍스트 프롬프트를 함께 이해하고, 주어진 컨텍스트에 기반하여 가장 적합한 결과를 생성합니다.
# 2. 컨텍스트를 활용한 추론 능력
# 제공된 코드(ocr_core_combine_3.py)를 보면 Gemini API를 호출할 때 다음과 같은 프롬프트를 사용합니다:
# "이 이미지는 한국 자동차 번호판입니다. 이미지에서 번호판 텍스트를 추출하고, JSON 형식으로 반환해줘. JSON 키는 'license_plate'야. 예시: {'license_plate': '12가3456'}. 번호판을 찾을 수 없으면 'license_plate'의 값은 null로 설정해줘."
# 이 프롬프트는 Gemini에게 "한국 자동차 번호판"이라는 명확한 컨텍스트를 제공하고, 원하는 결과 형식을 지정합니다. Gemini는 이 정보를 바탕으로 이미지 내의 글자들이 단순히 무작위의 문자가 아니라 특정 규칙(한국 번호판 패턴)을 따르는 텍스트임을 이해하고, 글자들의 시각적 특징뿐만 아니라 전체적인 패턴과 컨텍스트를 종합적으로 고려하여 추론합니다.
# 예를 들어, GCV는 '러'를 '2'로 인식하더라도 주변 글자와의 관계를 고려하지 않지만, Gemini는 번호판의 "숫자+한글+숫자" 패턴을 인지하고 '2'보다는 '러'가 더 자연스러운 결과임을 추론할 가능성이 높습니다.
# 3. 유연한 결과 반환 능력
# 또한, 제공된 코드에서는 Gemini에게 JSON 형식으로 결과를 반환하도록 요청합니다. 이는 Gemini가 단순히 텍스트를 나열하는 것을 넘어, 구조화된 형태로 답변을 생성할 수 있는 능력을 활용하는 것입니다. GCV가 반환하는 원시(raw) 텍스트를 후처리 로직으로 가공하는 것보다, 모델이 직접 의도된 형태의 결과를 제공하도록 유도하는 것이 더 정확한 결과를 얻는 데 유리합니다.
# 결론적으로, GCV가 문자 인식에 집중한다면, Gemini는 문맥 이해를 통한 정보 추출에 강점을 보입니다. 한국어 번호판과 같이 특정 규칙과 컨텍스트가 중요한 작업에서는 Gemini의 LLM 기반 접근 방식이 더 높은 정확도를 보여주는 것입니다.





import cv2
import numpy as np
import easyocr
import re
import json
import time
import os
import logging
from google.cloud import vision
import io

log = logging.getLogger("uvicorn")

_reader = None
_gcv_client = None

# --- ❗주의❗ 보안상 민감한 정보이므로 주의가 필요합니다. ---
# 이 파일은 Git에 함께 올라가게 되므로, 외부에 공개되지 않도록 주의하세요.
# SERVICE_ACCOUNT_KEY_PATH = "phonic-skyline-470005-t3-92381a8d7e79.json"


# 이 코드는 현재 실행 중인 .py 파일의 디렉터리 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
# 키 파일의 전체 경로를 구성합니다.
SERVICE_ACCOUNT_KEY_PATH = os.path.join(current_dir, "phonic-skyline-470005-t3-92381a8d7e79.json")
# -------------------------------------------------------------

def _get_reader():
    """EasyOCR Reader를 한 번만 초기화하여 반환합니다."""
    global _reader
    if _reader is None:
        log.info("easyocr.Reader init (cpu, ['ko','en']) ...")
        import easyocr
        _reader = easyocr.Reader(['ko','en'], gpu=False, verbose=False)
        log.info("easyocr.Reader ready")
    return _reader
    
def _get_gcv_client():
    """Google Cloud Vision 클라이언트를 한 번만 초기화하여 반환합니다."""
    global _gcv_client
    if _gcv_client is None:
        log.info("Google Cloud Vision Client init...")
        # 키 파일의 경로를 환경 변수에 설정하여 자동으로 인증합니다.
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
        _gcv_client = vision.ImageAnnotatorClient()
        log.info("Google Cloud Vision Client ready")
    return _gcv_client

# 1) 한국 번호판에 쓰이는 한글만 모아둔 문자열 (오인식 제외 목적)
allowed_letters = (
    "가나다라마"   # 자가용: 가~마
    "거너더러머버서어저"   # 자가용: 거~저
    "고노도로모보소오조"   # 자가용: 고~조
    "구누두루무부수우주"   # 자가용: 구~주
    "바사아자"     # 영업용: 바·사·아·자
    "배"           # 택배용
    "하허호"       # 렌터카용
)

LICENSE_PLATE_RE = re.compile(rf'(?P<p1>\d{{2,3}})(?P<kr>[{allowed_letters}])\s?(?P<p2>\d{{4}})')

# ==============================================================================
# --- 1단계: 빠르고 단순한 번호판 인식 로직 (EasyOCR) ---
# ==============================================================================

def preprocess_image_fast(image_path):
    """
    1단계: 이미지 파일을 그레이스케일, 리사이즈, 노이즈 제거 등 전처리합니다.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지 파일을 불러올 수 없습니다: {image_path}")

    # 이미지 가로 픽셀이 2000을 초과하면 비율을 유지하며 다운사이징
    height, width, _ = image.shape
    max_width = 2000
    if width > max_width:
        new_width = max_width
        new_height = int(height * (new_width / width))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image, thresh

def find_plate_roi(image, thresh, debug=False, save_base=None):
    """
    1단계: 윤곽선 탐색 후 번호판으로 보이는 사각형 영역을 추출합니다.
    - debug 모드에서 후보 영역을 시각화하여 저장합니다.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidate_regions = []
    image_width = image.shape[1]
    
    if debug and save_base:
        debug_image = image.copy()
    
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 0 and h > 0:
                aspect_ratio = w / h
                if (image_width * 0.15) < w < (image_width * 0.9):
                    if 3 < aspect_ratio < 6:
                        y_extended = max(0, int(y - h * 0.1))
                        h_extended = min(image.shape[0] - y_extended, int(h * 1.2))
                        roi = image[y_extended:y_extended+h_extended, x:x+w]
                        candidate_regions.append(roi)

                        if debug and save_base:
                            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            roi_filename = f"{os.path.splitext(save_base)[0]}_candidate_roi_{i}.jpg"
                            cv2.imwrite(roi_filename, roi)

    if debug and save_base:
        output_filename = f"{os.path.splitext(save_base)[0]}_rois_drawn.jpg"
        cv2.imwrite(output_filename, debug_image)

    return candidate_regions

def extract_text_from_image_fast(roi):
    """
    EasyOCR 결과를 한 줄/두 줄에 맞게 정렬·병합하고 한국 번호판 패턴으로 검증.
    """
    reader = _get_reader()
    result = reader.readtext(roi)
    ocr_results = []

    for bbox, text, conf in result:
        if not text:
            continue
        t = text.replace(" ", "")
        t = re.sub(r'[^0-9A-Za-z가-힣]', '', t)
        if not t:
            continue
        (x0,y0),(_, _),(x2,y2),(_,_) = bbox
        x_center = (x0 + x2) / 2.0
        y_center = (y0 + y2) / 2.0
        height   = abs(y2 - y0) + 1e-6
        ocr_results.append({"text": t, "confidence": float(conf),
                            "bbox": bbox, "x_center": x_center,
                            "y_center": y_center, "height": height})

    if not ocr_results:
        return None, 0.0, False

    def normalize_token(s: str) -> str:
        s = s.upper()
        s = s.replace("O", "0").replace("I", "1").replace("L", "1").replace("S", "5").replace("B", "8")
        s = re.sub(r'[^0-9가-힣]', '', s)
        return s

    for r in ocr_results:
        r["text"] = normalize_token(r["text"])

    mean_h = sum(r["height"] for r in ocr_results) / len(ocr_results)
    min_y  = min(r["y_center"] for r in ocr_results)
    max_y  = max(r["y_center"] for r in ocr_results)
    one_line = (max_y - min_y) <= (0.35 * mean_h)

    combined_text = ""
    if one_line:
        ocr_results.sort(key=lambda r: r["x_center"])
        combined_text = "".join(r["text"] for r in ocr_results)
    else:
        mid_y = (min_y + max_y) / 2.0
        top_line = [r for r in ocr_results if r["y_center"] <= mid_y]
        bot_line = [r for r in ocr_results if r["y_center"]  > mid_y]

        if not top_line or not bot_line:
            ocr_results.sort(key=lambda r: r["y_center"])
            half = max(1, len(ocr_results)//2)
            top_line, bot_line = ocr_results[:half], ocr_results[half:]

        top_line.sort(key=lambda r: r["x_center"])
        bot_line.sort(key=lambda r: r["x_center"])
        combined_text = "".join(r["text"] for r in top_line) + "".join(r["text"] for r in bot_line)

    if combined_text:
        m = LICENSE_PLATE_RE.search(combined_text)
        if m:
            plate = f"{m.group('p1')}{m.group('kr')}{m.group('p2')}"
            avg_conf = sum(r["confidence"] for r in ocr_results) / len(ocr_results)
            return plate, round(avg_conf, 3), True

    return None, 0.0, False

def recognize_plate_fast(image_path, debug=False):
    """
    1단계: 빠르고 단순한 OCR 워크플로우를 실행합니다.
    """
    last_checked_text = None
    try:
        save_base = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0])
        original_img, thresh = preprocess_image_fast(image_path)
        candidate_rois = find_plate_roi(original_img, thresh, debug=debug, save_base=save_base)
        
        if not candidate_rois:
            raise ValueError("번호판 후보 영역을 찾을 수 없습니다.")

        for i_roi, roi in enumerate(candidate_rois):
            plate_text, conf, is_valid = extract_text_from_image_fast(roi)
            
            if plate_text and is_valid:
                return {
                    "result": plate_text,
                    "confidence": conf,
                    "category": "car_number",
                    "success": True,
                    "source": "fast_ocr"
                }
            elif plate_text and not is_valid:
                last_checked_text = plate_text
        
        if last_checked_text:
            raise ValueError(f"OCR 결과에서 유효한 번호판을 찾지 못했습니다. 최종 실패 텍스트: {last_checked_text}")
        else:
            raise ValueError("OCR 결과에서 유효한 번호판을 찾지 못했습니다.")

    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

# ==============================================================================
# --- 2단계: Google Cloud Vision API를 활용한 정밀 인식 로직 ---
# ==============================================================================

def extract_korean_license_plate_gcv(image_path: str) -> dict:
    """
    Google Cloud Vision API를 사용하여 이미지에서 한국 자동차 번호판을 추출합니다.
    """
    gcv_client = _get_gcv_client()
    
    # 1. 이미지를 파일에서 읽어옵니다.
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # 2. 이미지 파일을 GCV API가 요구하는 형식으로 변환합니다.
    image = vision.Image(content=content)

    try:
        # 3. GCV API 호출
        response = gcv_client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            all_text = texts[0].description
            print('all_text from Google coud Vision : ', all_text)
            # 4. 추출된 텍스트를 번호판 패턴으로 검증합니다.
            lines = all_text.split('\n')
            for line in lines:
                m = LICENSE_PLATE_RE.search(line.replace(" ", ""))
                if m:
                    plate = f"{m.group('p1')}{m.group('kr')}{m.group('p2')}"
                    return {
                        "license_plate_number": plate,
                        "success": True,
                        "source": "gcv"
                    }

        # 5. 번호판을 찾지 못했을 경우
        return {
            "license_plate_number": None,
            "success": False,
            "source": "gcv",
            "error": "Google Cloud Vision API가 번호판 텍스트를 인식하지 못했습니다."
        }
    
    except Exception as e:
        return {
            "license_plate_number": None,
            "success": False,
            "source": "gcv",
            "error": f"Google Cloud Vision API 호출 중 오류 발생: {e}"
        }

# ==============================================================================
# --- 통합 워크플로우 ---
# ==============================================================================
def recognize_plate_combined(image_path, debug=False, reader=None, save_dir=None):
    """
    번호판 인식을 위한 통합 워크플로우.
    1) 빠른 방식 (EasyOCR) 먼저 시도
    2) 실패 시 정밀 방식 (Google Cloud Vision API) 시도
    """
    t0 = time.time()
    final_result = None
    
    # --- 1단계: 빠른 번호판 인식 시도 (EasyOCR) ---
    print("--- 1단계: 빠른 번호판 인식 시도 (EasyOCR) ---", flush=True)
    try:
        result_fast = recognize_plate_fast(image_path, debug=debug)
        if result_fast.get("success") and result_fast.get("confidence") >= 0.6:
            final_result = {
                "success": True,
                "plate_number": result_fast.get("result"),
                "confidence": result_fast.get("confidence"),
                "stage": "1단계 로직",
                "elapsed_sec": round(time.time() - t0, 2),
            }
            print("✅ 1단계에서 성공적으로 번호판을 인식했습니다", final_result.get("plate_number"), "confidence:", final_result.get("confidence"), flush=True)
        else:
            confidence_info = result_fast.get('confidence') if result_fast.get('success') else 'N/A'
            error_message = result_fast.get('error', '알 수 없는 오류')
            if result_fast.get('success') and confidence_info < 0.6:
                print(f"❌ 1단계 실패: Confidence가 0.6 미만 ({confidence_info})", flush=True)
            else:
                print(f"❌ 1단계 실패: {error_message}", flush=True)
    except Exception as e:
        print(f"❌ 1단계 예외: {e}", flush=True)
    
    # 1단계에서 성공하지 못했을 경우에만 2단계 GCV 실행
    if not final_result:
        # --- 2단계: Google Cloud Vision API를 활용한 정밀 인식 시도 ---
        print("--- 2단계: 정밀 번호판 인식 시도 (Google Cloud Vision API) ---", flush=True)
        try:
            gcv_result = extract_korean_license_plate_gcv(image_path)
            
            if gcv_result.get("success"):
                print("✅ 2단계 GCV에서 성공적으로 번호판을 인식했습니다.", flush=True)
                final_result = {
                    "success": True,
                    "plate_number": gcv_result.get("license_plate_number"),
                    "confidence": "gcv",
                    "stage": "2단계 gcv",
                    "elapsed_sec": round(time.time() - t0, 2),
                }
            else:
                print(f"❌ 2단계 실패: {gcv_result.get('error', '알 수 없는 오류')}", flush=True)
        except Exception as e:
            print(f"❌ 2단계 예외: {e}", flush=True)
    
    # 두 단계 모두 실패한 경우
    if not final_result:
        final_result = {
            "success": False,
            "plate_number": None,
            "confidence": None,
            "stage": "모두 실패",
            "elapsed_sec": round(time.time() - t0, 2),
        }

    return final_result

# ==============================================================================
# --- 메인 실행 블록 ---
# ==============================================================================
if __name__ == "__main__":
    # 테스트에 사용할 이미지 경로를 설정합니다.
    reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    image_dir = r"C:\01_Coding\250801_CAR_OCR_PHOTO\1_CAR_NO_OCR\test_samples"
    test_images = ['car1.jpg', 'car2.jpg', 'car3.jpg', 'car4.jpg', 'car5.jpg', 'car6.jpg', 'car7.jpg', 'car8.jpg', 'car9.jpg']
    # test_images = ['car5.jpg']
    
    debug_mode = True  #디버그 모드 설정 (사진저장)
    save_dir_base = r"C:\01_Coding\250801_CAR_OCR_PHOTO\1_CAR_NO_OCR\test_samples"

    print("--- 번호판 인식 테스트 시작 ---")
    for i, filename in enumerate(test_images):
        test_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(test_path):
            print(f"❌ 테스트 이미지가 존재하지 않습니다: {test_path}")
            continue
        else:
            print(f"🕒 번호판 인식 통합 워크플로우 시작... ({test_path})")
            
            # 통합 함수 호출
            result = recognize_plate_combined(test_path, debug=debug_mode, reader=reader, save_dir=save_dir_base)
            # result = recognize_plate_combined(test_path, debug=debug_mode, save_dir=save_dir_base)
            print("\n--- 최종 결과 ---")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("------------------")
    