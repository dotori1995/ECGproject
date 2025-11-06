import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import time
import wfdb
import librosa
from main import QDeepModel, CLASS_MAPPING, CLASS_NAMES

# --- [ 추론 함수 ] ---

def load_model(checkpoint_path, input_shape=(224, 224, 3), classes=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    저장된 모델 체크포인트를 로드합니다.
    
    Args:
        checkpoint_path: 체크포인트 파일 경로 (.pth)
        input_shape: 입력 이미지 크기
        classes: 클래스 수
        device: 사용할 디바이스 ('cuda' 또는 'cpu')
    
    Returns:
        로드된 모델
    """
    print(f"\n--- [모델 로드] ---")
    print(f"체크포인트 경로: {checkpoint_path}")
    print(f"사용 디바이스: {device}")
    
    # 모델 인스턴스 생성
    model = QDeepModel(input_shape=input_shape, classes=classes)
    
    # 체크포인트 로드
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"모델 로드 완료 (Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
    
    return model

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
            cqt = librosa.cqt(segment, sr=fs, hop_length=512, n_bins=84)
            cqt_magnitude = np.abs(cqt)
            cqt_log = librosa.amplitude_to_db(cqt_magnitude, ref=np.max)
            cqt_normalized = (cqt_log - cqt_log.min()) / (cqt_log.max() - cqt_log.min() + 1e-8)
            
            # 이미지 크기로 리사이즈
            cqt_tensor = torch.from_numpy(cqt_normalized).unsqueeze(0).unsqueeze(0)
            resized_img = F.interpolate(cqt_tensor, size=(img_shape[0], img_shape[1]), mode='bilinear', align_corners=False)
            resized_img = resized_img.squeeze().numpy()
            
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

def predict(model, X_data, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=32):
    """
    모델을 사용하여 예측을 수행합니다.
    
    Args:
        model: 학습된 모델
        X_data: 입력 데이터 (numpy array, shape: [N, H, W, C])
        device: 사용할 디바이스
        batch_size: 배치 크기
    
    Returns:
        예측 결과 (클래스 인덱스, 확률)
    """
    print("\n--- [예측 수행] ---")
    
    model.eval()
    
    # 데이터를 PyTorch 텐서로 변환 [N, H, W, C] -> [N, C, H, W]
    X_tensor = torch.FloatTensor(X_data).permute(0, 3, 1, 2).to(device)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        # 배치 단위로 처리
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            outputs = model(batch_X)
            probs = outputs.cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
    
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
    # 1. 체크포인트 파일 경로
    CHECKPOINT_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.pth"
    
    # 2. 추론할 데이터 파일 경로 (확장자 제외)
    FILE_BASE_PATH = "./data/data_0_1"  # 실제 데이터 경로로 수정하세요
    
    # 3. 모델 파라미터
    INPUT_SHAPE = (224, 224, 3)
    CLASSES = 3
    SEGMENT_LENGTH_SEC = 10
    OVERLAP_RATIO = 0.5
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------------
    
    # 1. 모델 로드
    try:
        model = load_model(
            checkpoint_path=CHECKPOINT_PATH,
            input_shape=INPUT_SHAPE,
            classes=CLASSES,
            device=DEVICE
        )
    except Exception as e:
        print(f"\n[치명적 오류] 모델 로드 실패: {e}")
        print("CHECKPOINT_PATH 경로와 파일 존재 여부를 확인하세요.")
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
    preds, probs = predict(
        model=model,
        X_data=X_data,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # 4. 결과 출력
    print_results(preds, probs, class_names=CLASS_NAMES)

