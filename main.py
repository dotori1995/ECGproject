!pip install wfdb
!pip install tensorflow

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import wfdb
import librosa

# --- [ 1. 모델 아키텍처 블록 정의 ] ---

def identity_block(X, f, filters):
    F1, F2 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def convolutional_block(X, f, filters, s=2):
    F1, F2 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(s, s), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# --- [ 2. QDeep 모델 아키텍처 (논문의 28 Layer 구조) ] ---

def QDeepModel(input_shape=(224, 224, 3), classes=3):
    """ 논문에 기술된 Q-Deep 모델 아키텍처를 구현합니다. """
    X_input = Input(input_shape)

    # 초기 처리 단계 (입력 크기 224x224x3 -> 56x56x64)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', padding='same')(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Layer 0 ~ Layer 5 (ResNet 구조)
    X = identity_block(X, 3, filters=[64, 64])       # Layer 0
    X = convolutional_block(X, f=3, filters=[128, 128], s=2)
    X = identity_block(X, 3, filters=[128, 128])     # Layer 1
    X = convolutional_block(X, f=3, filters=[256, 256], s=2)
    X = identity_block(X, 3, filters=[256, 256])     # Layer 2
    X = convolutional_block(X, f=3, filters=[256, 256], s=1)
    X = identity_block(X, 3, filters=[256, 256])     # Layer 3
    X = convolutional_block(X, f=3, filters=[512, 512], s=2)
    X = identity_block(X, 3, filters=[512, 512])     # Layer 4
    X = convolutional_block(X, f=3, filters=[512, 512], s=1)
    X = identity_block(X, 3, filters=[512, 512])     # Layer 5

    # 최종 단계
    X = AveragePooling2D((2, 2), name='avg_pool')(X)
    X = Flatten()(X)

    # 논문 그림에 따라 Dense Layer 추가
    X = Dense(32, activation='relu', name='fc_32')(X)
    X = Dense(16, activation='relu', name='fc_16')(X)

    X = Dense(classes, activation='softmax', name='fc_' + str(classes))(X)

    model = Model(inputs=X_input, outputs=X, name='QDeepModel')
    return model

# --- [ 3. 데이터 로딩 및 준비 (샘플 수 제한 추가) ] ---

# 클래스 레이블 매핑 (논문의 3개 클래스)
CLASS_MAPPING = {
    "non_atrial_fibrillation": 0,  # Non-Atrial Fibrillation
    "paroxysmal atrial fibrillation": 1,     # Paroxysmal Atrial Fibrillation
    "persistent atrial fibrillation": 2     # Persistent Atrial Fibrillation
}
CLASS_NAMES = list(CLASS_MAPPING.keys())

def load_data_from_single_file(file_base_path: str, img_shape=(224, 224, 3), segment_length_sec=10, overlap_ratio=0.5):
    """
    단일 atr, dat, hea 파일을 읽어서 CQT 변환 후 여러 세그먼트로 나누어 학습 데이터를 생성합니다.
    
    Args:
        file_base_path: .atr, .dat, .hea 파일의 기본 경로 (확장자 제외, 예: "data_0_1")
        img_shape: 이미지 크기 (height, width, channels)
        segment_length_sec: 각 세그먼트의 길이 (초)
        overlap_ratio: 세그먼트 간 겹침 비율 (0.0 ~ 1.0)
    """
    print("\n--- [3. 단일 파일 로드 및 처리] ---")
    
    # 1. WFDB 파일 읽기
    try:
        record = wfdb.rdrecord(file_base_path)
    except Exception as e:
        raise FileNotFoundError(f"파일을 읽을 수 없습니다: {file_base_path}\n오류: {e}")
    
    print(f"파일명: {file_base_path}")
    print(f"샘플링 레이트: {record.fs} Hz")
    print(f"신호 길이: {record.sig_len} 샘플 ({record.sig_len / record.fs:.2f} 초)")
    print(f"채널 수: {record.n_sig}")
    print(f"채널 이름: {record.sig_name}")
    
    # 2. 헤더 파일에서 클래스 레이블 읽기
    hea_file = file_base_path + '.hea'
    class_label = None
    class_label_id = None
    
    try:
        with open(hea_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('#'):
                    comment = line.strip()[1:].strip().lower()
                    # 클래스 레이블 매핑
                    if 'non atrial fibrillation' in comment or 'non_atrial_fibrillation' in comment:
                        class_label = "non_atrial_fibrillation"
                        class_label_id = CLASS_MAPPING[class_label]
                    elif 'paroxysmal atrial fibrillation' in comment:
                        class_label = "paroxysmal atrial fibrillation"
                        class_label_id = CLASS_MAPPING[class_label]
                    elif 'persistent atrial fibrillation' in comment:
                        class_label = "persistent atrial fibrillation"
                        class_label_id = CLASS_MAPPING[class_label]
    except Exception as e:
        print(f"경고: 헤더 파일에서 클래스 레이블을 읽을 수 없습니다: {e}")
        print("기본값으로 'non_atrial_fibrillation'을 사용합니다.")
        class_label = "non_atrial_fibrillation"
        class_label_id = CLASS_MAPPING[class_label]
    
    print(f"클래스 레이블: {class_label} (ID: {class_label_id})")
    
    # 3. 첫 번째 채널 데이터 추출 (일반적으로 I 또는 첫 번째 리드)
    ecg_signal = record.p_signal[:, 0]  # 첫 번째 채널 사용
    fs = record.fs
    
    # 4. 세그먼트로 나누기
    segment_length_samples = int(segment_length_sec * fs)
    overlap_samples = int(segment_length_samples * overlap_ratio)
    step_size = segment_length_samples - overlap_samples
    
    segments = []
    num_segments = (len(ecg_signal) - segment_length_samples) // step_size + 1
    
    print(f"\n세그먼트 길이: {segment_length_sec}초 ({segment_length_samples} 샘플)")
    print(f"겹침: {overlap_ratio * 100:.1f}% ({overlap_samples} 샘플)")
    print(f"예상 세그먼트 수: {num_segments}")
    
    # 5. 각 세그먼트를 CQT로 변환
    print("\n--- [4. CQT 변환 시작 (진행률 표시)] ---")
    start_time = time.time()
    
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length_samples
        
        if end_idx > len(ecg_signal):
            break
        
        segment = ecg_signal[start_idx:end_idx]
        
        # CQT 변환
        try:
            # librosa를 사용한 CQT 변환
            cqt = librosa.cqt(segment, sr=fs, hop_length=512, n_bins=84)
            cqt_magnitude = np.abs(cqt)
            
            # 로그 스케일 변환 (논문 방식)
            cqt_log = librosa.amplitude_to_db(cqt_magnitude, ref=np.max)
            
            # 정규화 (0~1 범위로)
            cqt_normalized = (cqt_log - cqt_log.min()) / (cqt_log.max() - cqt_log.min() + 1e-8)
            
            # 이미지 크기로 리사이즈
            resized_img = tf.image.resize(np.expand_dims(cqt_normalized, axis=-1),
                                          (img_shape[0], img_shape[1]), method='bilinear').numpy()[:, :, 0]
            
            # RGB 채널로 변환 (3채널)
            rgb_img = np.stack((resized_img, resized_img, resized_img), axis=-1)
            
            segments.append(rgb_img)
            
        except Exception as e:
            print(f"경고: 세그먼트 {i+1} CQT 변환 중 오류 발생: {e}. 건너뜁니다.")
            continue
        
        # 진행률 출력
        if (i + 1) % max(1, num_segments // 10) == 0 or (i + 1) == num_segments:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_segments * 100
            print(f"CQT 변환 진행: {i + 1:,}/{num_segments:,} ({progress:.1f}%) | 경과 시간: {elapsed:.2f}초", flush=True)
    
    if len(segments) == 0:
        raise ValueError("생성된 세그먼트가 없습니다. 파일 길이나 파라미터를 확인하세요.")
    
    # 6. 데이터 배열 생성
    X_data = np.array(segments, dtype=np.float32)
    Y_labels = to_categorical(np.full(len(segments), class_label_id), num_classes=len(CLASS_MAPPING))
    
    print("\n--- [4. 로드 완료] ---")
    print(f"데이터 크기 (X): {X_data.shape}, 레이블 크기 (Y): {Y_labels.shape}")
    print(f"생성된 세그먼트 수: {len(segments)}")
    print("----------------------------")
    
    return X_data, Y_labels

# --- [ 5. 학습 및 평가 함수 (이전과 동일) ] ---

def train_and_evaluate_model(model, X, Y, base_dir, epochs=60, batch_size=32):
    """
    논문의 70%/15%/15% 비율로 데이터를 분할하고 학습을 진행합니다.
    """
    print("\n--- [5. 데이터 분할 (논문 비율 70%/15%/15%)] ---")

    # 1. Test Set 분리 (15%)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )

    # 2. Train Set과 Validation Set 분리 (각각 70%와 15%에 근접하도록 분할)
    test_size_for_val = 0.15 / (1 - 0.15)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=test_size_for_val, random_state=42, stratify=Y_train_val
    )

    print(f"훈련 데이터 (Train): {X_train.shape[0]:,}개 (약 70%)")
    print(f"검증 데이터 (Validation): {X_val.shape[0]:,}개 (약 15%)")
    print(f"테스트 데이터 (Test): {X_test.shape[0]:,}개 (15%)")

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 콜백 설정
    checkpoint_path = os.path.join(base_dir, "best_qdeep_model_limited.keras")

    callbacks_list = [
        ModelCheckpoint(filepath=checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1),
    ]

    print(f"\n--- [6. 모델 학습 시작 (논문 기준 60 Epochs)] ---")
    # TensorFlow의 model.fit은 기본적으로 에포크별 진행률을 출력합니다.

    start_time = time.time()
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    end_time = time.time()
    print(f"\n총 학습 시간: {end_time - start_time:.2f} 초")

    # 베스트 모델 로드
    if os.path.exists(checkpoint_path):
        print(f"\n--- [7. 최적 가중치 로드] ---")
        model.load_weights(checkpoint_path)

    print("\n--- [8. 테스트 데이터로 모델 평가] ---")

    # 모델 평가 및 예측
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    Y_pred_probs = model.predict(X_test, verbose=0)
    Y_pred = np.argmax(Y_pred_probs, axis=1)
    Y_true = np.argmax(Y_test, axis=1)

    # 지표 계산
    test_accuracy = accuracy_score(Y_true, Y_pred)
    test_precision = precision_score(Y_true, Y_pred, average='weighted', zero_division=0)
    test_recall = recall_score(Y_true, Y_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(Y_true, Y_pred, average='weighted', zero_division=0)

    # 특이도(Specificity) 계산 (평균 특이도 사용)
    cm = confusion_matrix(Y_true, Y_pred)
    specificity_list = []
    for i in range(CLASSES):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    test_specificity = np.mean(specificity_list)

    print(f"\n\n===== 최종 테스트 결과 (논문 지표 기반) =====")
    print(f"테스트 손실 (Loss): {loss:.4f}")
    print(f"테스트 정확도 (Accuracy): {test_accuracy:.4f} (논문 목표: 0.98)")
    print(f"테스트 정밀도 (Precision): {test_precision:.4f} (논문 목표: 0.98)")
    print(f"테스트 민감도/재현율 (Sensitivity/Recall): {test_recall:.4f} (논문 목표: 0.98)")
    print(f"테스트 특이도 (Specificity, 평균): {test_specificity:.4f} (논문 목표: 0.97)")
    print(f"테스트 F1 점수 (F1 Score): {test_f1:.4f} (논문 목표: 0.98)")
    print("=============================================")

    return history

# --- [ 6. 메인 실행 ] ---

if __name__ == '__main__':

    # ⚠️ 1. 단일 atr, dat, hea 파일의 기본 경로 (확장자 제외)
    # 예: "data_0_1" (data_0_1.atr, data_0_1.dat, data_0_1.hea 파일이 있어야 함)
    FILE_BASE_PATH = "D:\git\온디바이스AI"

    # 2. 학습 아웃풋 (가중치 파일 저장) 경로
    PROJECT_BASE_DIR = Path("./qdeep_training_output_limited")

    # --- 학습 파라미터 ---
    EPOCHS = 60
    BATCH_SIZE = 32
    INPUT_SHAPE = (224, 224, 3)
    CLASSES = 3
    SEGMENT_LENGTH_SEC = 10  # 각 세그먼트의 길이 (초)
    OVERLAP_RATIO = 0.5  # 세그먼트 간 겹침 비율 (0.0 ~ 1.0)
    # -----------------------

    # 프로젝트 폴더 생성
    PROJECT_BASE_DIR.mkdir(exist_ok=True)

    # 1. 모델 인스턴스 생성
    model = QDeepModel(input_shape=INPUT_SHAPE, classes=CLASSES)
    print("--- [1. Q-Deep 모델 아키텍처 요약] ---")
    model.summary()

    # 2. 데이터 로드 (단일 파일에서 세그먼트 생성)
    try:
        X_data, Y_data = load_data_from_single_file(
            file_base_path=FILE_BASE_PATH,
            img_shape=INPUT_SHAPE,
            segment_length_sec=SEGMENT_LENGTH_SEC,
            overlap_ratio=OVERLAP_RATIO
        )
    except Exception as e:
        print(f"\n[치명적 오류] 데이터 로드 실패: {e}")
        print("FILE_BASE_PATH 경로와 파일 존재 여부를 확인하고 다시 시도하세요.")
        exit()

    if X_data.shape[0] == 0:
        print("[오류] 로드된 데이터가 0개입니다. 학습을 시작할 수 없습니다.")
        exit()

    # 3. 모델 학습 및 평가
    train_and_evaluate_model(
        model,
        X_data,
        Y_data,
        base_dir=PROJECT_BASE_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
