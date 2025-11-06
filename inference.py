import numpy as np
import os
from pathlib import Path
import time
import wfdb
import librosa
import onnxruntime as ort
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("경고: TensorFlow가 설치되지 않아 TensorFlow Lite 기능을 사용할 수 없습니다.")
from main import CLASS_MAPPING, CLASS_NAMES

# --- [ 추론 함수 ] ---

def load_onnx_model(onnx_path, device='cpu'):
    """
    ONNX 모델을 로드합니다.
    
    Args:
        onnx_path: ONNX 모델 파일 경로 (.onnx)
        device: 사용할 디바이스 ('cpu', 'cuda', 'tensorrt' 등)
    
    Returns:
        ONNX Runtime InferenceSession
    """
    print(f"\n--- [ONNX 모델 로드] ---")
    print(f"ONNX 모델 경로: {onnx_path}")
    print(f"사용 디바이스: {device}")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {onnx_path}")
    
    # ONNX Runtime 세션 옵션 설정
    providers = []
    if device == 'cuda':
        # CUDA 사용 가능 여부 확인
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except:
            providers = ['CPUExecutionProvider']
            print("경고: CUDA를 사용할 수 없어 CPU로 실행합니다.")
    else:
        providers = ['CPUExecutionProvider']
    
    # 세션 생성
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 입력/출력 정보 출력
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    
    print(f"모델 로드 완료")
    print(f"  입력 이름: {input_name}, 크기: {input_shape}")
    print(f"  출력 이름: {output_name}, 크기: {output_shape}")
    print(f"  실행 제공자: {session.get_providers()}")
    
    return session

def load_tflite_model(tflite_path):
    """
    TensorFlow Lite 모델을 로드합니다.
    
    Args:
        tflite_path: TensorFlow Lite 모델 파일 경로 (.tflite)
    
    Returns:
        TensorFlow Lite Interpreter
    """
    if not TFLITE_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다. pip install tensorflow")
    
    print(f"\n--- [TensorFlow Lite 모델 로드] ---")
    print(f"TFLite 모델 경로: {tflite_path}")
    
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TensorFlow Lite 모델 파일을 찾을 수 없습니다: {tflite_path}")
    
    # TensorFlow Lite Interpreter 생성
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 입력/출력 정보 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"모델 로드 완료")
    print(f"  입력 이름: {input_details[0]['name']}, 크기: {input_details[0]['shape']}, 타입: {input_details[0]['dtype']}")
    print(f"  출력 이름: {output_details[0]['name']}, 크기: {output_details[0]['shape']}, 타입: {output_details[0]['dtype']}")
    
    return interpreter

def preprocess_single_file(file_base_path: str, img_shape=(224, 224, 3), segment_length_sec=10, overlap_ratio=0.5):
    """
    단일 atr, dat, hea 파일을 읽어서 CQT 변환 후 여러 세그먼트로 나누어 추론 데이터를 생성합니다.
    
    Args:
        file_base_path: .atr, .dat, .hea 파일의 기본 경로 (확장자 제외)
        img_shape: 이미지 크기 (height, width, channels)
        segment_length_sec: 각 세그먼트의 길이 (초)
        overlap_ratio: 세그먼트 간 겹침 비율 (0.0 ~ 1.0)
    
    Returns:
        세그먼트 배열 (numpy array)
    """
    print("\n--- [데이터 전처리] ---")
    
    # 1. WFDB 파일 읽기
    try:
        record = wfdb.rdrecord(file_base_path)
    except Exception as e:
        raise FileNotFoundError(f"파일을 읽을 수 없습니다: {file_base_path}\n오류: {e}")
    
    print(f"파일명: {file_base_path}")
    print(f"샘플링 레이트: {record.fs} Hz")
    print(f"신호 길이: {record.sig_len} 샘플 ({record.sig_len / record.fs:.2f} 초)")
    
    # 2. 첫 번째 채널 데이터 추출
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    
    # 3. 세그먼트로 나누기
    segment_length_samples = int(segment_length_sec * fs)
    overlap_samples = int(segment_length_samples * overlap_ratio)
    step_size = segment_length_samples - overlap_samples
    
    segments = []
    num_segments = (len(ecg_signal) - segment_length_samples) // step_size + 1
    
    print(f"세그먼트 길이: {segment_length_sec}초 ({segment_length_samples} 샘플)")
    print(f"겹침: {overlap_ratio * 100:.1f}% ({overlap_samples} 샘플)")
    print(f"예상 세그먼트 수: {num_segments}")
    print("\n--- [CQT 변환 시작] ---")
    
    start_time = time.time()
    
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length_samples
        
        if end_idx > len(ecg_signal):
            break
        
        segment = ecg_signal[start_idx:end_idx]
        
        # CQT 변환
        try:
            # CQT 파라미터 계산 (main.py와 동일한 로직)
            nyquist = fs / 2
            max_octaves = np.log2(nyquist / 27.5)
            n_bins = min(84, int(12 * max_octaves * 0.9))
            hop_length = max(32, int(fs / 4))
            
            cqt = librosa.cqt(segment, sr=fs, hop_length=hop_length, n_bins=n_bins, fmin=27.5)
            cqt_magnitude = np.abs(cqt)
            cqt_log = librosa.amplitude_to_db(cqt_magnitude, ref=np.max)
            cqt_normalized = (cqt_log - cqt_log.min()) / (cqt_log.max() - cqt_log.min() + 1e-8)
            
            # 이미지 크기로 리사이즈 (scipy를 사용)
            from scipy.ndimage import zoom
            current_shape = cqt_normalized.shape
            zoom_factors = (img_shape[0] / current_shape[0], img_shape[1] / current_shape[1])
            resized_img = zoom(cqt_normalized, zoom_factors, order=1)  # bilinear interpolation
            
            # RGB 채널로 변환
            rgb_img = np.stack((resized_img, resized_img, resized_img), axis=-1)
            segments.append(rgb_img)
            
        except Exception as e:
            print(f"경고: 세그먼트 {i+1} CQT 변환 중 오류 발생: {e}. 건너뜁니다.")
            continue
        
        # 진행률 출력
        if (i + 1) % max(1, num_segments // 10) == 0 or (i + 1) == num_segments:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_segments * 100
            print(f"CQT 변환 진행: {i+1:,}/{num_segments:,} ({progress:.1f}%) | 경과 시간: {elapsed:.2f}초", flush=True)
    
    if len(segments) == 0:
        raise ValueError("생성된 세그먼트가 없습니다.")
    
    X_data = np.array(segments, dtype=np.float32)
    print(f"\n전처리 완료: {X_data.shape[0]}개 세그먼트 생성")
    
    return X_data

def predict_onnx(session, X_data, batch_size=32):
    """
    ONNX 모델을 사용하여 예측을 수행합니다.
    
    Args:
        session: ONNX Runtime InferenceSession
        X_data: 입력 데이터 (numpy array, shape: [N, H, W, C])
        batch_size: 배치 크기
    
    Returns:
        예측 결과 (클래스 인덱스, 확률)
    """
    print("\n--- [ONNX 예측 수행] ---")
    
    # 입력/출력 이름 가져오기
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 데이터를 ONNX 형식으로 변환 [N, H, W, C] -> [N, C, H, W]
    # numpy array를 float32로 변환
    X_data_float32 = X_data.astype(np.float32)
    X_data_chw = np.transpose(X_data_float32, (0, 3, 1, 2))  # [N, H, W, C] -> [N, C, H, W]
    
    all_preds = []
    all_probs = []
    
    # 배치 단위로 처리
    for i in range(0, len(X_data_chw), batch_size):
        batch_X = X_data_chw[i:i+batch_size]
        
        # ONNX Runtime으로 추론
        outputs = session.run([output_name], {input_name: batch_X})
        probs = outputs[0]  # 첫 번째 출력
        
        preds = np.argmax(probs, axis=1)
        
        all_preds.extend(preds)
        all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"예측 완료: {len(all_preds)}개 세그먼트")
    
    return all_preds, all_probs

def predict_tflite(interpreter, X_data, batch_size=1):
    """
    TensorFlow Lite 모델을 사용하여 예측을 수행합니다.
    
    Args:
        interpreter: TensorFlow Lite Interpreter
        X_data: 입력 데이터 (numpy array, shape: [N, H, W, C])
        batch_size: 배치 크기 (TFLite는 보통 1)
    
    Returns:
        예측 결과 (클래스 인덱스, 확률)
    """
    print("\n--- [TensorFlow Lite 예측 수행] ---")
    
    # 입력/출력 정보 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 데이터를 TFLite 형식으로 변환 [N, H, W, C] -> [N, C, H, W]
    X_data_float32 = X_data.astype(np.float32)
    X_data_chw = np.transpose(X_data_float32, (0, 3, 1, 2))  # [N, H, W, C] -> [N, C, H, W]
    
    all_preds = []
    all_probs = []
    
    # TFLite는 배치 크기가 1인 경우가 많으므로 하나씩 처리
    for i in range(len(X_data_chw)):
        input_data = X_data_chw[i:i+1]  # 배치 차원 유지 [1, C, H, W]
        
        # 입력 데이터 설정
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # 추론 실행
        interpreter.invoke()
        
        # 출력 가져오기
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probs = output_data[0]  # 배치 차원 제거
        
        pred = np.argmax(probs)
        
        all_preds.append(pred)
        all_probs.append(probs)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"예측 완료: {len(all_preds)}개 세그먼트")
    
    return all_preds, all_probs

def print_results(preds, probs, class_names=None):
    """
    예측 결과를 출력합니다.
    
    Args:
        preds: 예측된 클래스 인덱스 배열
        probs: 예측 확률 배열
        class_names: 클래스 이름 리스트
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    print("\n--- [예측 결과] ---")
    
    # 각 세그먼트별 결과
    print("\n[세그먼트별 예측 결과]")
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        print(f"세그먼트 {i+1}: {class_names[pred]} (확률: {prob[pred]:.4f})")
        print(f"  - 전체 확률 분포: {dict(zip(class_names, prob))}")
    
    # 전체 통계
    print("\n[전체 통계]")
    unique, counts = np.unique(preds, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        percentage = count / len(preds) * 100
        print(f"  {class_names[cls_idx]}: {count}개 ({percentage:.1f}%)")
    
    # 최종 예측 (다수결)
    final_pred = unique[np.argmax(counts)]
    final_prob = np.mean(probs[:, final_pred])
    print(f"\n[최종 예측 (다수결)]")
    print(f"  클래스: {class_names[final_pred]}")
    print(f"  평균 확률: {final_prob:.4f}")
    print(f"  지지 세그먼트: {counts[np.argmax(counts)]}/{len(preds)}개")

# --- [ 메인 실행 ] ---

if __name__ == '__main__':
    
    # ⚠️ 설정 파라미터
    # 모델 타입 선택: 'onnx' 또는 'tflite'
    MODEL_TYPE = 'onnx'  # 'onnx' 또는 'tflite'
    
    # 1. 모델 파일 경로
    ONNX_MODEL_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.onnx"
    TFLITE_MODEL_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.tflite"
    
    # 2. 추론할 데이터 파일 경로 (확장자 제외)
    FILE_BASE_PATH = "./data/data_0_1"  # 실제 데이터 경로로 수정하세요
    
    # 3. 모델 파라미터
    INPUT_SHAPE = (224, 224, 3)
    SEGMENT_LENGTH_SEC = 10
    OVERLAP_RATIO = 0.5
    BATCH_SIZE = 32
    DEVICE = 'cpu'  # 'cpu' 또는 'cuda' (ONNX Runtime만 해당)
    # -----------------------
    
    # 1. 모델 로드
    if MODEL_TYPE == 'onnx':
        try:
            model = load_onnx_model(
                onnx_path=ONNX_MODEL_PATH,
                device=DEVICE
            )
        except Exception as e:
            print(f"\n[치명적 오류] ONNX 모델 로드 실패: {e}")
            print("ONNX_MODEL_PATH 경로와 파일 존재 여부를 확인하세요.")
            print("ONNX 모델이 없다면 convert_to_onnx.py를 먼저 실행하세요.")
            exit()
    elif MODEL_TYPE == 'tflite':
        try:
            model = load_tflite_model(
                tflite_path=TFLITE_MODEL_PATH
            )
        except Exception as e:
            print(f"\n[치명적 오류] TensorFlow Lite 모델 로드 실패: {e}")
            print("TFLITE_MODEL_PATH 경로와 파일 존재 여부를 확인하세요.")
            print("TensorFlow Lite 모델이 없다면 convert_to_tflite.py를 먼저 실행하세요.")
            exit()
    else:
        print(f"[오류] 지원하지 않는 모델 타입: {MODEL_TYPE}")
        print("MODEL_TYPE을 'onnx' 또는 'tflite'로 설정하세요.")
        exit()
    
    # 2. 데이터 전처리
    try:
        X_data = preprocess_single_file(
            file_base_path=FILE_BASE_PATH,
            img_shape=INPUT_SHAPE,
            segment_length_sec=SEGMENT_LENGTH_SEC,
            overlap_ratio=OVERLAP_RATIO
        )
    except Exception as e:
        print(f"\n[치명적 오류] 데이터 전처리 실패: {e}")
        print("FILE_BASE_PATH 경로와 파일 존재 여부를 확인하세요.")
        exit()
    
    if X_data.shape[0] == 0:
        print("[오류] 전처리된 데이터가 0개입니다.")
        exit()
    
    # 3. 예측 수행
    if MODEL_TYPE == 'onnx':
        preds, probs = predict_onnx(
            session=model,
            X_data=X_data,
            batch_size=BATCH_SIZE
        )
    elif MODEL_TYPE == 'tflite':
        preds, probs = predict_tflite(
            interpreter=model,
            X_data=X_data,
            batch_size=1  # TFLite는 보통 배치 크기 1
        )
    
    # 4. 결과 출력
    print_results(preds, probs, class_names=CLASS_NAMES)

