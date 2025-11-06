import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import wfdb
import librosa

# Windows 터미널 출력 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- [ 1. 모델 아키텍처 블록 정의 ] ---

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, f, filters):
        super(IdentityBlock, self).__init__()
        F1, F2 = filters
        padding = f // 2  # kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=(f, f), stride=(1, 1), padding=padding)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=(f, f), stride=(1, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(F2)

    def forward(self, X):
        X_shortcut = X
        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X)
        X = self.conv2(X)
        X = self.bn2(X)
        X = X + X_shortcut
        X = F.relu(X)
        return X

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, f, filters, s=2):
        super(ConvolutionalBlock, self).__init__()
        F1, F2 = filters
        # PyTorch에서는 stride > 1일 때 padding='same'을 지원하지 않으므로 수동 계산
        padding = f // 2  # kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=(f, f), stride=(s, s), padding=padding)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=(f, f), stride=(1, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv_shortcut = nn.Conv2d(in_channels, F2, kernel_size=(1, 1), stride=(s, s), padding=0)
        self.bn_shortcut = nn.BatchNorm2d(F2)

    def forward(self, X):
        X_shortcut = X
        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X)
        X = self.conv2(X)
        X = self.bn2(X)
        X_shortcut = self.conv_shortcut(X_shortcut)
        X_shortcut = self.bn_shortcut(X_shortcut)
        X = X + X_shortcut
        X = F.relu(X)
        return X

# --- [ 2. QDeep 모델 아키텍처 (논문의 28 Layer 구조) ] ---

class QDeepModel(nn.Module):
    """ 논문에 기술된 Q-Deep 모델 아키텍처를 구현합니다. """
    def __init__(self, input_shape=(224, 224, 3), classes=3):
        super(QDeepModel, self).__init__()

        # 초기 처리 단계 (입력 크기 224x224x3 -> 56x56x64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Layer 0 ~ Layer 5 (ResNet 구조)
        # Layer 0: identity block (64 -> 64)
        self.identity_block0 = IdentityBlock(64, 3, [64, 64])

        # Convolutional block (64 -> 128)
        self.conv_block1 = ConvolutionalBlock(64, 3, [128, 128], s=2)
        # Layer 1: identity block (128 -> 128)
        self.identity_block1 = IdentityBlock(128, 3, [128, 128])

        # Convolutional block (128 -> 256)
        self.conv_block2 = ConvolutionalBlock(128, 3, [256, 256], s=2)
        # Layer 2: identity block (256 -> 256)
        self.identity_block2 = IdentityBlock(256, 3, [256, 256])

        # Convolutional block (256 -> 256, stride=1)
        self.conv_block3 = ConvolutionalBlock(256, 3, [256, 256], s=1)
        # Layer 3: identity block (256 -> 256)
        self.identity_block3 = IdentityBlock(256, 3, [256, 256])

        # Convolutional block (256 -> 512)
        self.conv_block4 = ConvolutionalBlock(256, 3, [512, 512], s=2)
        # Layer 4: identity block (512 -> 512)
        self.identity_block4 = IdentityBlock(512, 3, [512, 512])

        # Convolutional block (512 -> 512, stride=1)
        self.conv_block5 = ConvolutionalBlock(512, 3, [512, 512], s=1)
        # Layer 5: identity block (512 -> 512)
        self.identity_block5 = IdentityBlock(512, 3, [512, 512])

        # 최종 단계
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 논문 그림에 따라 Dense Layer 추가
        self.fc_32 = nn.Linear(512, 32)
        self.fc_16 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, classes)

    def forward(self, X):
        # 초기 처리
        X = self.conv1(X)
        X = self.bn_conv1(X)
        X = F.relu(X)
        X = self.maxpool(X)

        # Layer 0 ~ Layer 5
        X = self.identity_block0(X)
        X = self.conv_block1(X)
        X = self.identity_block1(X)
        X = self.conv_block2(X)
        X = self.identity_block2(X)
        X = self.conv_block3(X)
        X = self.identity_block3(X)
        X = self.conv_block4(X)
        X = self.identity_block4(X)
        X = self.conv_block5(X)
        X = self.identity_block5(X)

        # 최종 단계
        X = self.avgpool(X)
        X = self.flatten(X)
        X = F.relu(self.fc_32(X))
        X = F.relu(self.fc_16(X))
        X = F.softmax(self.fc_out(X), dim=1)

        return X

# --- [ 3. 데이터 로딩 및 준비 (샘플 수 제한 추가) ] ---

# 클래스 레이블 매핑 (논문의 3개 클래스)
CLASS_MAPPING = {
    "non_atrial_fibrillation": 0,  # Non-Atrial Fibrillation
    "paroxysmal atrial fibrillation": 1,     # Paroxysmal Atrial Fibrillation
    "persistent atrial fibrillation": 2     # Persistent Atrial Fibrillation
}
CLASS_NAMES = list(CLASS_MAPPING.keys())

def to_categorical(y, num_classes):
    """PyTorch용 one-hot encoding 함수"""
    y = np.array(y, dtype=np.int64)
    categorical = np.zeros((len(y), num_classes))
    categorical[np.arange(len(y)), y] = 1
    return categorical

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

    # 5. CQT 파라미터 사전 계산 (루프 밖에서 한 번만 계산)
    nyquist = fs / 2
    max_octaves = np.log2(nyquist / 27.5)
    n_bins = min(84, int(12 * max_octaves * 0.9))  # 안전 마진 포함
    # hop_length를 샘플링 레이트에 맞게 조정 (너무 크면 느림)
    # 200 Hz의 경우 hop_length를 더 작게 설정 (예: fs/4 = 50)
    hop_length = max(32, int(fs / 4))  # 최소 32, 샘플링 레이트의 1/4

    print(f"\n--- [4. CQT 변환 시작 (진행률 표시)] ---")
    print(f"CQT 파라미터: n_bins={n_bins}, hop_length={hop_length}, fmin=27.5 Hz")
    start_time = time.time()

    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length_samples

        if end_idx > len(ecg_signal):
            break

        segment = ecg_signal[start_idx:end_idx]

        # CQT 변환
        try:
            # librosa를 사용한 CQT 변환 (파라미터는 이미 계산됨)
            cqt = librosa.cqt(segment, sr=fs, hop_length=hop_length, n_bins=n_bins, fmin=27.5)
            cqt_magnitude = np.abs(cqt)

            # 로그 스케일 변환 (논문 방식)
            cqt_log = librosa.amplitude_to_db(cqt_magnitude, ref=np.max)

            # 정규화 (0~1 범위로)
            cqt_normalized = (cqt_log - cqt_log.min()) / (cqt_log.max() - cqt_log.min() + 1e-8)

            # 이미지 크기로 리사이즈 (PyTorch 방식)
            cqt_tensor = torch.from_numpy(cqt_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            resized_img = F.interpolate(cqt_tensor, size=(img_shape[0], img_shape[1]), mode='bilinear', align_corners=False)
            resized_img = resized_img.squeeze().numpy()  # [H, W]

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

# --- [ 5. 학습 및 평가 함수 ] ---

def train_and_evaluate_model(model, X, Y, base_dir, epochs=60, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    논문의 70%/15%/15% 비율로 데이터를 분할하고 학습을 진행합니다.
    """
    print(f"\n--- [5. 데이터 분할 (논문 비율 70%/15%/15%)] ---")
    print(f"사용 디바이스: {device}")

    # 1. Test Set 분리 (15%)
    Y_argmax = np.argmax(Y, axis=1)  # stratify를 위한 클래스 인덱스
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y_argmax
    )

    # 2. Train Set과 Validation Set 분리 (각각 70%와 15%에 근접하도록 분할)
    Y_train_val_argmax = np.argmax(Y_train_val, axis=1)
    test_size_for_val = 0.15 / (1 - 0.15)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=test_size_for_val, random_state=42, stratify=Y_train_val_argmax
    )

    print(f"훈련 데이터 (Train): {X_train.shape[0]:,}개 (약 70%)")
    print(f"검증 데이터 (Validation): {X_val.shape[0]:,}개 (약 15%)")
    print(f"테스트 데이터 (Test): {X_test.shape[0]:,}개 (15%)")

    # 데이터를 PyTorch 텐서로 변환 및 DataLoader 생성
    # PyTorch는 [N, C, H, W] 형식을 사용하므로 [N, H, W, C] -> [N, C, H, W]로 변환
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2).to(device)
    Y_test_tensor = torch.FloatTensor(Y_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델을 디바이스로 이동
    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 체크포인트 경로
    checkpoint_path = os.path.join(base_dir, "best_qdeep_model_limited.pth")
    best_val_loss = float('inf')

    print(f"\n--- [6. 모델 학습 시작 (논문 기준 60 Epochs)] ---")

    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(batch_Y.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(batch_Y.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  -> 베스트 모델 저장 (Val Loss: {val_loss:.4f})")

    end_time = time.time()
    print(f"\n총 학습 시간: {end_time - start_time:.2f} 초")

    # 베스트 모델 로드
    if os.path.exists(checkpoint_path):
        print(f"\n--- [7. 최적 가중치 로드] ---")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"로드된 모델의 Val Loss: {checkpoint['val_loss']:.4f}")

    print("\n--- [8. 테스트 데이터로 모델 평가] ---")

    # 모델 평가 및 예측
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(batch_Y.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    Y_pred = np.array(all_preds)
    Y_true = np.array(all_labels)

    # 지표 계산
    test_accuracy = accuracy_score(Y_true, Y_pred)
    test_precision = precision_score(Y_true, Y_pred, average='weighted', zero_division=0)
    test_recall = recall_score(Y_true, Y_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(Y_true, Y_pred, average='weighted', zero_division=0)

    # 특이도(Specificity) 계산 (평균 특이도 사용)
    # labels 인자를 명시하여 모든 클래스에 대한 confusion matrix를 생성
    cm = confusion_matrix(Y_true, Y_pred, labels=list(CLASS_MAPPING.values()))
    specificity_list = []
    num_classes = len(CLASS_MAPPING)
    for i in range(num_classes):
        # 각 클래스에 대해 TN, FP를 계산하여 특이도를 구함
        # cm[i, i]는 True Positive for class i
        # cm[:, i].sum()은 class i로 예측된 모든 경우 (TP + FP)
        # cm[i, :].sum()은 실제 class i인 모든 경우 (TP + FN)
        # cm.sum()은 전체 샘플 수

        # True Negative (TN): class i도 아니고, class i로 예측되지도 않은 경우
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))

        # False Positive (FP): class i로 잘못 예측된 다른 클래스들
        fp = np.sum(cm[:, i]) - cm[i, i]

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    test_specificity = np.mean(specificity_list)

    print(f"\n\n===== 최종 테스트 결과 (논문 지표 기반) =====")
    print(f"테스트 손실 (Loss): {test_loss:.4f}")
    print(f"테스트 정확도 (Accuracy): {test_accuracy:.4f} (논문 목표: 0.98)")
    print(f"테스트 정밀도 (Precision): {test_precision:.4f} (논문 목표: 0.98)")
    print(f"테스트 민감도/재현율 (Sensitivity/Recall): {test_recall:.4f} (논문 목표: 0.98)")
    print(f"테스트 특이도 (Specificity, 평균): {test_specificity:.4f} (논문 목표: 0.97)")
    print(f"테스트 F1 점수 (F1 Score): {test_f1:.4f} (논문 목표: 0.98)")
    print("=============================================")

    return history

# --- [ 6. 메인 실행 ] ---

if __name__ == '__main__':

    # ☢☢ 1. 단일 atr, dat, hea 파일의 기본 경로 (확장자 제외)
    # 예: "data_0_1" 또는 "./data/data_0_1" (data_0_1.atr, data_0_1.dat, data_0_1.hea 파일이 있어야 함)
    FILE_BASE_PATH = "./content/data/data_0_1"  # 실제 데이터 경로로 수정하세요

    # 2. 학습 아웃풋 (가중치 파일 저장) 경로
    PROJECT_BASE_DIR = Path("./content/qdeep_training_output_limited")

    # --- 학습 파라미터 ---
    EPOCHS = 60
    BATCH_SIZE = 32
    INPUT_SHAPE = (224, 224, 3)
    CLASSES = 3
    SEGMENT_LENGTH_SEC = 10  # 각 세그먼트의 길이 (초)
    OVERLAP_RATIO = 0.5  # 세그먼트 간 겹침 비율 (0.0 ~ 1.0)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------------

    # 프로젝트 폴더 생성
    PROJECT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 모델 인스턴스 생성
    model = QDeepModel(input_shape=INPUT_SHAPE, classes=CLASSES)
    print("--- [1. Q-Deep 모델 아키텍처 요약] ---")
    print(model)

    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")

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
        batch_size=BATCH_SIZE,
        device=DEVICE
    )