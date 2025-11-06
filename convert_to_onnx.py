import torch
import torch.nn as nn
import torch.onnx
import os
from pathlib import Path
from main import QDeepModel, CLASS_MAPPING

def convert_pytorch_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape=(224, 224, 3),
    classes=3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    PyTorch 모델을 ONNX 형식으로 변환합니다.
    
    Args:
        checkpoint_path: PyTorch 체크포인트 파일 경로 (.pth)
        output_path: ONNX 파일 저장 경로 (.onnx)
        input_shape: 입력 이미지 크기 (height, width, channels)
        classes: 클래스 수
        device: 사용할 디바이스 ('cuda' 또는 'cpu')
    """
    print("\n--- [PyTorch to ONNX 변환] ---")
    print(f"체크포인트 경로: {checkpoint_path}")
    print(f"출력 경로: {output_path}")
    print(f"입력 크기: {input_shape}")
    print(f"사용 디바이스: {device}")
    
    # 1. 모델 인스턴스 생성
    print("\n[1/5] 모델 인스턴스 생성 중...")
    model = QDeepModel(input_shape=input_shape, classes=classes)
    
    # 2. 체크포인트 로드
    print("[2/5] 체크포인트 로드 중...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"모델 로드 완료 (Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
    
    # 3. 더미 입력 생성 (PyTorch는 [N, C, H, W] 형식)
    print("[3/5] 더미 입력 생성 중...")
    batch_size = 1
    dummy_input = torch.randn(batch_size, input_shape[2], input_shape[0], input_shape[1]).to(device)
    print(f"더미 입력 크기: {dummy_input.shape}")
    
    # 4. ONNX로 변환
    print("[4/5] ONNX 변환 중...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 입력/출력 이름 지정
    input_names = ['input_image']  # 입력 이름
    output_names = ['output_probabilities']  # 출력 이름
    
    # 동적 축 설정 (배치 크기)
    dynamic_axes = {
        'input_image': {0: 'batch_size'},  # 배치 크기를 동적으로 설정
        'output_probabilities': {0: 'batch_size'}
    }
    
    try:
        torch.onnx.export(
            model,                          # 모델
            dummy_input,                    # 더미 입력
            output_path,                    # 출력 파일 경로
            export_params=True,              # 학습된 파라미터 포함
            opset_version=13,               # ONNX opset 버전 (안정적인 버전)
            do_constant_folding=True,        # 상수 폴딩 최적화
            input_names=input_names,        # 입력 이름
            output_names=output_names,      # 출력 이름
            dynamic_axes=dynamic_axes,      # 동적 축 설정
            verbose=False                   # 상세 출력 비활성화
        )
        print(f"✓ ONNX 변환 완료: {output_path}")
    except Exception as e:
        print(f"✗ ONNX 변환 실패: {e}")
        raise
    
    # 5. 변환된 파일 검증
    print("[5/5] ONNX 파일 검증 중...")
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB 단위
        print(f"✓ ONNX 파일 생성 완료")
        print(f"  파일 크기: {file_size:.2f} MB")
        print(f"  파일 경로: {output_path}")
        
        # ONNX 모델 로드 테스트
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX 모델 검증 성공")
        except ImportError:
            print("⚠ onnx 패키지가 설치되지 않아 검증을 건너뜁니다.")
            print("  설치: pip install onnx")
        except Exception as e:
            print(f"⚠ ONNX 모델 검증 중 경고: {e}")
    else:
        raise FileNotFoundError(f"ONNX 파일이 생성되지 않았습니다: {output_path}")
    
    print("\n--- [변환 완료] ---")
    print(f"원본 모델: {checkpoint_path}")
    print(f"ONNX 모델: {output_path}")
    print("\nONNX 모델 사용 방법:")
    print("  import onnxruntime as ort")
    print("  session = ort.InferenceSession('모델경로.onnx')")
    print("  outputs = session.run(None, {'input_image': input_data})")
    
    return output_path

if __name__ == '__main__':
    # 설정 파라미터
    CHECKPOINT_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.pth"
    OUTPUT_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.onnx"
    INPUT_SHAPE = (224, 224, 3)
    CLASSES = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 변환 실행
    try:
        convert_pytorch_to_onnx(
            checkpoint_path=CHECKPOINT_PATH,
            output_path=OUTPUT_PATH,
            input_shape=INPUT_SHAPE,
            classes=CLASSES,
            device=DEVICE
        )
    except Exception as e:
        print(f"\n[치명적 오류] 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

