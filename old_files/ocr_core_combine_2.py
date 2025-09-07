# ocr_core_combined.py
# ocr_core_combine_2.py :  2단계 없앰. 1단계만 사용.

import cv2
import numpy as np
import easyocr
import re
import json
import time
import os

# OCR 엔진 초기화 (한글 + 영문)
# EasyOCR을 한 번만 초기화하여 자원 낭비를 방지합니다.
reader = easyocr.Reader(['ko', 'en'], gpu=False)

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

# 1줄 번호판 정규식 패턴 (띄어쓰기 허용)
license_plate_pattern_one_line = re.compile(r'^\d{2,3}[가-힣]\s?\d{4}$')
# 2줄 번호판 정규식 패턴
license_plate_pattern_two_line = re.compile(r'^[가-힣]{2}\d{2}[가-힣]\d{4}$')

# 최종 결과 검증을 위한 단순화된 패턴: 뒤에서 5번째 한 글자는 allowed_letters, 뒤 4자리는 숫자
final_simplified_pattern = re.compile(rf'[{allowed_letters}]\d{{4}}$')

# 허용 문자 패턴 (번호판에 나올 수 있는 숫자+한글)
korean_chars = '가나다라마바사아자거너버서어저고노도로모보소오조구누두루무부수우주하허호배육해공국합'
allow_re = re.compile(f"[0-9{''.join(korean_chars)}]+")

# ocr_core_combine_1.py
import logging
log = logging.getLogger("uvicorn")

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        log.info("easyocr.Reader init (cpu, ['en','ko']) ...")
        import easyocr
        _reader = easyocr.Reader(['en','ko'], gpu=False, verbose=False)
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

    image = cv2.resize(image, (800, 600))
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
    1단계: OCR 엔진으로 텍스트를 추출하고 유효성 검사를 수행합니다.
    - EasyOCR이 분리된 텍스트('28노', '7587')를 반환할 경우 합치는 로직을 포함합니다.
    """
    result = reader.readtext(roi)
    ocr_results = []
    
    for bbox, text, confidence in result:
        text = text.replace(" ", "")
        text = re.sub(r'[^0-9가-힣]', '', text)
        if text:
            ocr_results.append({'text': text, 'confidence': confidence, 'bbox': bbox})
    
    if len(ocr_results) == 1:
        text_combo = ocr_results[0]['text']
        if license_plate_pattern_one_line.match(text_combo) or license_plate_pattern_two_line.match(text_combo):
            if final_simplified_pattern.search(text_combo):
                return text_combo, round(float(ocr_results[0]['confidence']), 3), True
            else:
                return text_combo, round(float(ocr_results[0]['confidence']), 3), False

    elif len(ocr_results) > 1:
        combined_text = "".join([res['text'] for res in ocr_results])
        combined_confidence = sum([res['confidence'] for res in ocr_results]) / len(ocr_results)
        if license_plate_pattern_one_line.match(combined_text) or license_plate_pattern_two_line.match(combined_text):
            if final_simplified_pattern.search(combined_text):
                return combined_text, round(float(combined_confidence), 3), True
            else:
                return combined_text, round(float(combined_confidence), 3), False
                
    return None, 0.0, False

def recognize_plate_fast(image_path, debug=False):
    """
    1단계: 빠르고 단순한 OCR 워크플로우를 실행합니다.
    """
    last_checked_text = None
    try:
        start_time = time.time()
        save_base = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0])
        original_img, thresh = preprocess_image_fast(image_path)
        candidate_rois = find_plate_roi(original_img, thresh, debug=debug, save_base=save_base)
        
        if not candidate_rois:
            raise ValueError("번호판 후보 영역을 찾을 수 없습니다.")

        for i_roi, roi in enumerate(candidate_rois):
            plate_text, conf, is_valid = extract_text_from_image_fast(roi)
            if plate_text and is_valid:
                duration = round(time.time() - start_time, 2)
                return {
                    "result": plate_text,
                    "confidence": conf,
                    "time": duration,
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
# --- 2단계: 기울기 보정 기반의 정밀 OCR 로직 (ocr_core_8_onlyOCR_3_angle.py에서 가져옴) ---
# ==============================================================================

def center_of_bbox(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / 4, sum(ys) / 4

def get_group_bbox(group):
    all_x = [p[0] for char in group for p in char['bbox']]
    all_y = [p[1] for char in group for p in char['bbox']]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    return [x_min, y_min, x_max, y_max]

def get_rotation_angle(group):
    if len(group) < 2:
        return 0
    xs = np.array([c["cx"] for c in group])
    ys = np.array([c["cy"] for c in group])
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    numerator = np.sum((xs - mean_x) * (ys - mean_y))
    denominator = np.sum((xs - mean_x)**2)
    if denominator == 0:
        return 0
    slope = numerator / denominator
    angle = np.arctan(slope) * 180 / np.pi
    return angle

def get_regex_similarity_score(text):
    no_space = text.replace(" ", "")
    if license_plate_pattern_one_line.match(no_space):
        return 2.0
    if license_plate_pattern_two_line.match(no_space):
        return 2.0
    if len(no_space) in [7, 8] and no_space.isdigit():
        return 1.5
    return 0.0

def score_candidate(text, confs):
    base_conf = np.mean(confs) if confs else 0.0
    regex_score = get_regex_similarity_score(text)
    return base_conf + regex_score

def recognize_plate_robust(image_path, debug=False, save_dir=None):
    """
    2단계: 기울기 보정 및 문자 그룹화를 통해 번호판을 인식하는 정밀 OCR 워크플로우를 실행합니다.
    """
    overall_start = time.perf_counter()
    img_ori = cv2.imread(image_path)
    if img_ori is None:
        return {"error": f"이미지를 불러올 수 없습니다: {image_path}", "success": False}
    
    save_base = None
    if debug and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_base = os.path.join(save_dir, os.path.splitext(filename)[0])
    
    # 전처리
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_img = clahe.apply(gray)
    blurred_img = cv2.GaussianBlur(preprocessed_img, ksize=(5, 5), sigmaX=0)
    
    # EasyOCR 전체 이미지 OCR
    all_ocr_results = reader.readtext(blurred_img)
    
    # 전체 OCR 결과에 대한 즉시 검증
    for bbox, text, conf in all_ocr_results:
        cleaned_text = text.replace(" ", "")
        if (license_plate_pattern_one_line.match(cleaned_text) or 
            license_plate_pattern_two_line.match(cleaned_text)):
            if final_simplified_pattern.search(cleaned_text.replace(" ", "")):
                return {
                    "result": cleaned_text,
                    "confidence": conf,
                    "score": score_candidate(cleaned_text, [conf]),
                    "time": round(time.perf_counter() - overall_start, 2),
                    "category": "car_number",
                    "success": True,
                    "source": "robust_single_ocr"
                }

    # OCR 결과 필터링 및 문자 정보 가공
    all_chars = []
    for bbox, text, conf in all_ocr_results:
        cleaned = text.replace(" ", "")
        if conf < 0.3 or not allow_re.search(cleaned):
            continue
        cx, cy = center_of_bbox(bbox)
        xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        all_chars.append({"bbox": bbox, "text": cleaned, "conf": conf, "cx": cx, "cy": cy, "w": w, "h": h})
    
    # 문자 그룹 기반 후보 생성
    char_groups = []
    used = set()
    for i, c in enumerate(all_chars):
        if i in used: continue
        group = [c]
        for j, d in enumerate(all_chars):
            if i == j or j in used: continue
            if abs(c["cy"] - d["cy"]) < max(c["h"], d["h"]) * 0.8 and abs(c["cx"] - d["cx"]) < (c["w"] + d["w"]) * 5:
                group.append(d)
                used.add(j)
        if len(group) >= 2:
            combined_text = "".join([g["text"] for g in group])
            if len(re.findall(r'\d', combined_text)) >= 4:
                char_groups.append(group)
                used.add(i)

    candidates = []
    # 1차 OCR 그룹을 후보로 추가
    for group in char_groups:
        texts = [g["text"] for g in group]
        confs = [g["conf"] for g in group]
        combined_text = "".join(texts)
        score = score_candidate(combined_text, confs)
        if score > 0:
            candidates.append({"text": combined_text, "confidence": np.mean(confs) if confs else 0.0,
                               "score": score, "source": "char_group_original"})
    
    # 후보 그룹 기울기 보정 및 재 OCR
    for gi, group in enumerate(char_groups):
        angle = get_rotation_angle(group)
        if abs(angle) > 10:
            x_min, y_min, x_max, y_max = get_group_bbox(group)
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max = min(img_ori.shape[1], int(x_max))
            y_max = min(img_ori.shape[0], int(y_max))
            group_img = img_ori[y_min:y_max, x_min:x_max]
            if group_img.size == 0: continue
            h, w = group_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated_img = cv2.warpAffine(group_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            re_ocr_results = reader.readtext(rotated_img)
            if re_ocr_results:
                re_ocr_texts = [r[1] for r in re_ocr_results]
                re_ocr_confs = [r[2] for r in re_ocr_results]
                re_ocr_combined_text = "".join(re_ocr_texts).replace(" ", "")
                score = score_candidate(re_ocr_combined_text, re_ocr_confs)
                if score > 0:
                    candidates.append({"text": re_ocr_combined_text, "confidence": np.mean(re_ocr_confs) if re_ocr_confs else 0.0,
                                       "score": score, "source": "char_group_corrected"})

    # 최적 후보 선택
    if not candidates:
        return {"error": "유효한 번호판 후보를 찾지 못했습니다.", "success": False}
    best = max(candidates, key=lambda c: c["score"])

    # 최종 결과 형식 검증
    final_result_text = best["text"].replace(" ", "")
    if final_simplified_pattern.search(final_result_text):
        return {
            "result": final_result_text,
            "confidence": best.get("confidence", 0.0),
            "score": best["score"],
            "time": round(time.perf_counter() - overall_start, 2),
            "category": "car_number",
            "success": True,
            "source": best.get("source", "")
        }
    else:
        return {
            "error": f"최종 결과 '{final_result_text}'가 유효한 번호판 형식에 맞지 않습니다.",
            "success": False,
            "time": round(time.perf_counter() - overall_start, 2),
            "category": "car_number_failed"
        }

# ==============================================================================
# --- 최종 통합 함수: 두 로직을 순차적으로 실행 ---
# ==============================================================================
# def recognize_plate_combined(image_path, debug=False, save_dir=None):
#     reader = _get_reader()
#     """
#     번호판 인식을 위한 통합 워크플로우를 실행합니다.
#     1. 빠르고 단순한 방법으로 먼저 시도합니다.
#     2. 실패할 경우, 기울기 보정 기능이 있는 정밀한 방법으로 다시 시도합니다.
#     """
#     print("--- 1단계: 빠른 번호판 인식 시도 ---")
#     result_fast = recognize_plate_fast(image_path, debug=debug)
    
#     if result_fast.get("success"):
#         print("✅ 1단계에서 성공적으로 번호판을 인식했습니다.")
#         return result_fast
#     else:
#         print(f"❌ 1단계 실패: {result_fast.get('error', '알 수 없는 오류')}")
#         print("--- 2단계: 정밀 번호판 인식 시도 ---")
#         result_robust = recognize_plate_robust(image_path, debug=debug, save_dir=save_dir)
        
#         if result_robust.get("success"):
#             print("✅ 2단계에서 성공적으로 번호판을 인식했습니다.")
#             return result_robust
#         else:
#             print(f"❌ 2단계 실패: {result_robust.get('error', '알 수 없는 오류')}")
#             return {
#                 "error": "두 단계 모두 번호판 인식이 실패했습니다.",
#                 "success": False
#             }

def recognize_plate_combined(image_path, debug=False, save_dir=None):
    """
    번호판 인식을 위한 통합 워크플로우.
    1) 빠른 방식 먼저 시도
    2) 실패 시 정밀 방식 시도 (이때에만 easyocr Reader 초기화)
    """
    import time, os
    t0 = time.time()

    # --- 1단계: 빠른 인식 ---
    print("--- 1단계: 빠른 번호판 인식 시도 ---", flush=True)
    try:
        result_fast = recognize_plate_fast(image_path, debug=debug)
        if result_fast.get("success"):
            print("✅ 1단계에서 성공적으로 번호판을 인식했습니다.", flush=True)
            result_fast["stage"] = "fast"
            result_fast["elapsed_sec"] = round(time.time() - t0, 2)
            return result_fast
        else:
            print(f"❌ 1단계 실패: {result_fast.get('error', '알 수 없는 오류')}", flush=True)
    except Exception as e:
        print(f"❌ 1단계 예외: {e}", flush=True)

    # --- 2단계: 정밀 인식 (이때만 Reader 준비) ---
    # print("--- 2단계: 정밀 번호판 인식 시도 ---", flush=True)
    # try:
    #     # 여기서만 무거운 리더 초기화 → 초기 지연/메모리 사용을 줄임
    #     reader = _get_reader()
    #     print("easyocr.Reader 준비 완료", flush=True)
    # except Exception as e:
    #     return {
    #         "success": False,
    #         "error": f"easyocr 초기화 실패: {e}",
    #         "elapsed_sec": round(time.time() - t0, 2)
    #     }

    # try:
    #     # robust가 reader 인자를 지원하면 전달, 아니면 TypeError 시 빼고 호출
    #     try:
    #         result_robust = recognize_plate_robust(
    #             image_path, debug=debug, save_dir=save_dir, reader=reader
    #         )
    #     except TypeError:
    #         result_robust = recognize_plate_robust(
    #             image_path, debug=debug, save_dir=save_dir
    #         )
    # except Exception as e:
    #     return {
    #         "success": False,
    #         "error": f"2단계 처리 중 예외: {e}",
    #         "elapsed_sec": round(time.time() - t0, 2)
    #     }

    # if result_robust.get("success"):
    #     print("✅ 2단계에서 성공적으로 번호판을 인식했습니다.", flush=True)
    #     result_robust["stage"] = "robust"
    #     result_robust["elapsed_sec"] = round(time.time() - t0, 2)
    #     return result_robust
    # else:
    #     print(f"❌ 2단계 실패: {result_robust.get('error', '알 수 없는 오류')}", flush=True)
    #     return {
    #         "success": False,
    #         "error": "두 단계 모두 번호판 인식이 실패했습니다.",
    #         "elapsed_sec": round(time.time() - t0, 2)
    #     }


# ==============================================================================
# --- 메인 실행 블록 ---
# ==============================================================================
if __name__ == "__main__":
    # 테스트에 사용할 이미지 경로를 설정합니다.
    # 사용자의 환경에 맞게 경로를 수정하세요.
    
    image_dir = r"C:\01_Coding\250801_CAR_OCR_PHOTO\1_CAR_NO_OCR\test_samples"
    # test_images = ['car1.jpg', 'car2.jpg', 'car3.jpg', 'car4.jpg', 'car5.jpg', 'car6.jpg', 'car7.jpg', 'car8.jpg', 'car9.jpg']
    test_images = ['car1.jpg', 'car2.jpg']
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
            result = recognize_plate_combined(test_path, debug=debug_mode, save_dir=save_dir_base)
            print("\n--- 최종 결과 ---")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("------------------")
    