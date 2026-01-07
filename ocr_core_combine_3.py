# ocr_core_combined.py
# ocr_core_combine_2.py :  2단계 없앰. 1단계만 사용.
# ocr_core_combine_3.py :  2단계 삭제. gemini 로직 추가  // 가로픽셀 2천 다운사이징 로직 추가.

import cv2
import numpy as np
import easyocr
import re
import json
import time
import os
import base64
import requests

# OCR 엔진 초기화 (한글 + 영문)
# EasyOCR을 한 번만 초기화하여 자원 낭비를 방지합니다.
# reader = easyocr.Reader(['ko', 'en'], gpu=False)

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

# 1줄 번호판 정규식 패턴 (띄어쓰기 허용)
# license_plate_pattern_one_line = re.compile(r'^\d{2,3}[가-힣]\s?\d{4}$')
# # 2줄 번호판 정규식 패턴
# license_plate_pattern_two_line = re.compile(r'^[가-힣]{2}\d{2}[가-힣]\d{4}$')
# # 최종 결과 검증을 위한 단순화된 패턴: 뒤에서 5번째 한 글자는 allowed_letters, 뒤 4자리는 숫자
# final_simplified_pattern = re.compile(rf'[{allowed_letters}]\d{{4}}$')
# # 허용 문자 패턴 (번호판에 나올 수 있는 숫자+한글)
# korean_chars = '가나다라마바사아자거너버서어저고노도로모보소오조구누두루무부수우주하허호배육해공국합'
# allow_re = re.compile(f"[0-9{''.join(korean_chars)}]+")



# ocr_core_combine_1.py
import logging
log = logging.getLogger("uvicorn")

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        log.info("easyocr.Reader init (cpu, ['ko','en']) ...")
        import easyocr
        _reader = easyocr.Reader(['ko','en'], gpu=False, verbose=False)
        log.info("easyocr.Reader ready")
    return _reader
# ==============================================================================
# --- 1단계: 빠르고 단순한 번호판 인식 로직 (ocr_core_9_1_simple_fast4.py에서 가져옴) ---
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
        # 비율을 유지하며 새로운 가로/세로 길이 계산
        new_width = max_width
        new_height = int(height * (new_width / width))
        # cv2.resize 함수를 사용하여 이미지 크기 조정 (보간법은 INTER_AREA 사용)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 이 부분은 원래 리사이즈 로직을 그대로 사용 (필요에 따라 주석 처리 또는 제거 가능)
    # 기존 코드에서 800x600으로 고정 리사이즈하는 부분.
    # 위 다운사이징 로직을 넣었기 때문에 이 부분은 주석 처리하거나 목적에 맞게 수정해야 합니다.
    # 여기서는 고정 리사이즈 부분을 제거하고 다운사이징된 이미지를 그대로 사용하겠습니다.
    
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
    result = reader.readtext(roi)  # [(bbox, text, conf), ...]
    ocr_results = []

    # 1) 정리
    for bbox, text, conf in result:
        if not text:
            continue
        t = text.replace(" ", "")
        t = re.sub(r'[^0-9A-Za-z가-힣]', '', t)  # 일단 영문도 허용 후 후처리에서 치환
        if not t:
            continue
        # 중심 좌표/크기
        (x0,y0),(_, _),(x2,y2),(_,_) = bbox
        x_center = (x0 + x2) / 2.0
        y_center = (y0 + y2) / 2.0
        height   = abs(y2 - y0) + 1e-6
        ocr_results.append({"text": t, "confidence": float(conf),
                            "bbox": bbox, "x_center": x_center,
                            "y_center": y_center, "height": height})

    print("ocr_results", ocr_results)

    if not ocr_results:
        return None, 0.0, False

    # 2) 문자인식 흔한 보정
    def normalize_token(s: str) -> str:
        s = s.upper()
        s = s.replace("O", "0").replace("I", "1").replace("L", "1").replace("S", "5").replace("B", "8")
        s = re.sub(r'[^0-9가-힣]', '', s)  # 숫자/한글만 유지
        return s

    for r in ocr_results:
        r["text"] = normalize_token(r["text"])

    # 3) 한 줄 / 두 줄 판별
    mean_h = sum(r["height"] for r in ocr_results) / len(ocr_results)
    min_y  = min(r["y_center"] for r in ocr_results)
    max_y  = max(r["y_center"] for r in ocr_results)
    one_line = (max_y - min_y) <= (0.35 * mean_h)

    combined_text = ""
    if one_line:
        # 한 줄 → 좌→우
        ocr_results.sort(key=lambda r: r["x_center"])
        combined_text = "".join(r["text"] for r in ocr_results)
    else:
        # 두 줄 → y로 위/아래 분리, 각 줄에서 좌→우
        mid_y = (min_y + max_y) / 2.0
        top_line = [r for r in ocr_results if r["y_center"] <= mid_y]
        bot_line = [r for r in ocr_results if r["y_center"]  > mid_y]

        # 예외: 잘 안 나뉘면 y기준 정렬 후 반으로 나눔
        if not top_line or not bot_line:
            ocr_results.sort(key=lambda r: r["y_center"])
            half = max(1, len(ocr_results)//2)
            top_line, bot_line = ocr_results[:half], ocr_results[half:]

        top_line.sort(key=lambda r: r["x_center"])
        bot_line.sort(key=lambda r: r["x_center"])
        combined_text = "".join(r["text"] for r in top_line) + "".join(r["text"] for r in bot_line)

    # 4) ←←← 중요: 검증/반환은 if/else '밖'에서 수행 (전문 or 부분 매치)
    if combined_text:
        print("combined_text:", repr(combined_text))  # 디버그
        m = LICENSE_PLATE_RE.search(combined_text)    # 부분 매치 허용(예: '356어873951' → '356어8739')
        print("regex_match:", bool(m))                # 디버그
        if m:
            plate = f"{m.group('p1')}{m.group('kr')}{m.group('p2')}"  # 공백 제거 정규화
            avg_conf = sum(r["confidence"] for r in ocr_results) / len(ocr_results)
            return plate, round(avg_conf, 3), True

    # 항상 3-튜플 반환
    return None, 0.0, False



def recognize_plate_fast(image_path, debug=False):
    
    """
    1단계: 빠르고 단순한 OCR 워크플로우를 실행합니다.
    """
    last_checked_text = None
    try:
        # start_time = time.time()
        save_base = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0])
        original_img, thresh = preprocess_image_fast(image_path)
        candidate_rois = find_plate_roi(original_img, thresh, debug=debug, save_base=save_base)
        print('len(candidate_rois)', len(candidate_rois))
        
        if not candidate_rois:
            raise ValueError("번호판 후보 영역을 찾을 수 없습니다.")

        for i_roi, roi in enumerate(candidate_rois):
            ret = extract_text_from_image_fast(roi)
            if not ret:
                print(i_roi, "번째  결과 없음(None) → 다음 ROI")
                continue

            plate_text, conf, is_valid = ret
            print(i_roi, "번째  text:", plate_text)

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
        
        
        
        
        
        
        
        
        

# Gemini API를 호출하여 이미지에서 한국 자동차 번호판을 추출하고 검증하는 함수
def extract_korean_license_plate_gemini(image_path: str) -> dict:
    """
    이미지에서 한국 자동차 번호판을 추출하고 유효성을 검증합니다.
    Args:
        image_path (str): 번호판이 포함된 이미지 파일의 경로.
    Returns:
        dict: 추출 결과, 성공 여부, 소스 정보 등을 담은 딕셔너리.
              성공 시: {"license_plate_number": "12가3456", "success": True, "source": "gemini"}
              실패 시: {"license_plate_number": None, "success": False, "source": "gemini", "error": "에러 메시지"}
    """
    # Canvas 환경에서는 API 키를 자동으로 제공합니다.
    api_key = "AIzaSyDZ7PC6WA6xE86fH2OGX-XslImFEvBcKM8"  #gemini api key


    

    # 1. 한국 번호판에 쓰이는 한글 문자열 (오인식 방지 목적)
    allowed_letters = (
        "가나다라마"      # 자가용: 가~마
        "거너더러머버서어저"      # 자가용: 거~저
        "고노도로모보소오조"      # 자가용: 고~조
        "구누두루무부수우주"      # 자가용: 구~주
        "바사아자"        # 영업용: 바·사·아·자
        "배"              # 택배용
        "하허호"          # 렌터카용
        "육해공국합"      # 특수용도
    )

    # 2. 번호판 정규식 패턴 (띄어쓰기 허용)
    license_plate_pattern = re.compile(
        rf'^\d{{2,3}}[{allowed_letters}]\s?\d{{4}}$'
    )

    try:
        # # 이미지를 base64로 인코딩합니다.
        # with open(image_path, "rb") as image_file:
        #     base64_image = base64.b64encode(image_file.read()).decode("utf-8")


        # 1. 이미지를 OpenCV로 읽어옵니다.
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 파일을 불러올 수 없습니다: {image_path}")

        # 2. 가로 픽셀이 2000을 초과하면 비율에 맞게 다운사이징합니다.
        height, width, _ = image.shape
        max_width = 2000
        if width > max_width:
            new_width = max_width
            new_height = int(height * (new_width / width))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 3. 수정된 이미지를 메모리 버퍼에 저장한 후 base64로 인코딩합니다.
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode("utf-8")



        # 4. Gemini API 호출을 위한 페이로드를 구성합니다.
        # 응답을 JSON 형식으로 받도록 요청합니다.
        prompt_text = (
            "이 이미지는 한국 자동차 번호판입니다. "
            "이미지에서 번호판 텍스트를 추출하고, "
            "JSON 형식으로 반환해줘. "
            "JSON 키는 'license_plate'야. "
            "예시: {'license_plate': '12가3456'}. "
            "번호판을 찾을 수 없으면 'license_plate'의 값은 null로 설정해줘."
        )

        # 1. 페이로드 구성 (v1 정식 버전 규격)
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        # 2. API URL (v1beta 버전)
        # model_name = "gemini-1.5-flash"
        model_name = "gemini-2.5-flash"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        
        # 상세한 에러 확인을 위해 아래 코드를 추가하는 것이 좋습니다.
        if response.status_code != 200:
            print(f"에러 발생 코드: {response.status_code}")
            print(f"에러 내용: {response.text}")
            
        response.raise_for_status() 
        json_response = response.json()
        
        # JSON 응답에서 텍스트를 추출하고 파싱합니다.
        raw_text = json_response['candidates'][0]['content']['parts'][0]['text']
        parsed_data = json.loads(raw_text)
        plate_text = parsed_data.get("license_plate")

        # 3. 추출된 번호판 텍스트를 정규식으로 검증합니다.
        if plate_text and license_plate_pattern.match(plate_text):
            # 성공적으로 번호판을 추출하고 검증했을 경우
            return {
                "license_plate_number": plate_text.replace(" ", ""),
                "success": True,
                "source": "gemini"
            }
        else:
            # 추출된 번호판이 없거나 정규식과 일치하지 않을 경우
            return {
                "license_plate_number": None,
                "success": False,
                "source": "gemini",
                "error": "번호판을 찾을 수 없거나 형식이 올바르지 않습니다."
            }

    except requests.exceptions.RequestException as e:
        # API 호출 중 네트워크 또는 HTTP 오류 발생 시
        return {
            "license_plate_number": None,
            "success": False,
            "source": "gemini",
            "error": f"Gemini API 호출 중 오류 발생: {e}"
        }
    except (KeyError, json.JSONDecodeError) as e:
        # 응답 JSON 파싱 중 오류 발생 시
        return {
            "license_plate_number": None,
            "success": False,
            "source": "gemini",
            "error": f"Gemini API 응답 파싱 중 오류 발생: {e}"
        }
    except FileNotFoundError:
        # 이미지 파일을 찾을 수 없을 경우
        return {
            "license_plate_number": None,
            "success": False,
            "source": "gemini",
            "error": f"파일을 찾을 수 없습니다: {image_path}"
        }
    except Exception as e:
        # 그 외 예상치 못한 오류 발생 시
        return {
            "license_plate_number": None,
            "success": False,
            "source": "gemini",
            "error": f"예상치 못한 오류 발생: {e}"
        }

# 수정된 코드
# def recognize_plate_combined(image_path, debug=False, save_dir=None):
def recognize_plate_combined(image_path, debug=False, reader=None, save_dir=None):
# def recognize_plate_combined(image_path, debug=False, save_dir=None):
    """
    번호판 인식을 위한 통합 워크플로우.
    1) 빠른 방식 먼저 시도
    2) 실패 시 정밀 방식 시도 (Gemini 로직)
    """
    import time, os
    t0 = time.time()
    
    final_result = None
    
    '''
    # --- 1단계: 빠른 인식 ---
    print("--- 1단계: 빠른 번호판 인식 시도 ---", flush=True)
    final_result = None
    try:
        result_fast = recognize_plate_fast(image_path, debug=debug)
        # 1단계 성공 여부 및 confidence를 확인
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
            # success가 False이거나 confidence가 0.6 미만일 경우
            confidence_info = result_fast.get('confidence') if result_fast.get('success') else 'N/A'
            error_message = result_fast.get('error', '알 수 없는 오류')
            if result_fast.get('success') and confidence_info < 0.6:
                print(f"❌ 1단계 실패: Confidence가 0.6 미만 ({confidence_info})", flush=True)
            else:
                print(f"❌ 1단계 실패: {error_message}", flush=True)
    except Exception as e:
        print(f"❌ 1단계 예외: {e}", flush=True)
    '''
    
    
    final_result =  ''  #임시로 값 할당함
    # 1단계에서 성공하지 못했을 경우에만 2단계 Gemini 실행
    if not final_result:
        # --- 2단계: Gemini를 활용한 정밀 인식 시도 ---
        print("--- 2단계 Gemini 정밀 번호판 인식 시도 ---", flush=True)
        try:
            gemini_result = extract_korean_license_plate_gemini(image_path)
            
            if gemini_result.get("success"):
                print("✅ 2단계 Gemini에서 성공적으로 번호판을 인식했습니다.", flush=True)
                final_result = {
                    "success": True,
                    "plate_number": gemini_result.get("license_plate_number"), # 'license_plate_number' -> 'plate_number'로 변경
                    "confidence": "gemini",
                    "stage": "2단계 gemini",
                    "elapsed_sec": round(time.time() - t0, 2),
                }
            else:
                print(f"❌ 2단계 실패: {gemini_result.get('error', '알 수 없는 오류')}", flush=True)
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
    # 사용자의 환경에 맞게 경로를 수정하세요.
    
    reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    
    
    image_dir = r"C:\01_Coding\250801_CAR_OCR_PHOTO\1_CAR_NO_OCR\test\error"
    # test_images = ['car1.jpg', 'car2.jpg', 'car3.jpg', 'car4.jpg', 'car5.jpg', 'car6.jpg', 'car7.jpg', 'car8.jpg', 'car9.jpg']
    test_images = ['img1.jpg', 'img2.jpg']
    # test_images = ['KakaoTalk_20250911_163213445_01.jpg']
    # test_images = ['KakaoTalk_20250910_104158897_04.jpg']
    debug_mode = True  #디버그 모드 설정 (사진저장)
    save_dir_base = r"C:\01_Coding\250801_CAR_OCR_PHOTO\1_CAR_NO_OCR\test\error"

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
    