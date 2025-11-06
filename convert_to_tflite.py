"""
ONNX 모델을 TensorFlow Lite로 변환하는 스크립트

주의: 현재 Windows 환경에서는 의존성 문제로 변환이 어려울 수 있습니다.
대안: ONNX Runtime을 직접 사용하세요 (inference.py에서 MODEL_TYPE='onnx' 설정)

필요한 패키지 설치 옵션:
  옵션 ①: pip install onnx2tf ai-edge-litert tensorflow
  옵션 ②: pip install onnx==1.14.0 onnx-tf tensorflow

현재 환경에서는 ONNX Runtime만 사용하는 것을 권장합니다.
"""
import os
import sys
from pathlib import Path

# Windows 터미널 출력 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def convert_onnx_to_tflite(onnx_path, output_path):
    """
    ONNX 모델을 TensorFlow Lite로 변환합니다.
    
    Args:
        onnx_path: ONNX 모델 파일 경로 (.onnx)
        output_path: TensorFlow Lite 파일 저장 경로 (.tflite)
    """
    print("\n--- [ONNX to TensorFlow Lite 변환] ---")
    print(f"ONNX 모델 경로: {onnx_path}")
    print(f"출력 경로: {output_path}")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {onnx_path}")
    
    try:
        # 변환 방법 시도 순서: onnx2tf -> onnx-tf
        use_onnx2tf = False
        conversion_method = None
        
        # 방법 1: onnx2tf 시도 (최신, 더 안정적)
        try:
            from onnx2tf import convert
            use_onnx2tf = True
            conversion_method = "onnx2tf"
            print("onnx2tf를 사용합니다.")
        except ImportError as e1:
            print(f"onnx2tf import 실패: {e1}")
            # 방법 2: onnx-tf 시도
            try:
                import onnx
                from onnx_tf.backend import prepare
                use_onnx2tf = False
                conversion_method = "onnx-tf"
                print("onnx-tf를 사용합니다.")
            except ImportError as e2:
                print(f"onnx-tf import 실패: {e2}")
                # 방법 3: 직접 ONNX Runtime을 사용하고 TensorFlow Lite는 건너뛰기
                print("\n[경고] ONNX -> TensorFlow 변환 도구를 사용할 수 없습니다.")
                print("대안: ONNX Runtime을 직접 사용하세요 (inference.py에서 MODEL_TYPE='onnx' 설정)")
                raise ImportError(f"onnx2tf 또는 onnx-tf가 필요합니다.\n  onnx2tf 오류: {e1}\n  onnx-tf 오류: {e2}\n\n해결 방법:\n  1. pip install onnx2tf tensorflow (추천)\n  2. 또는 ONNX Runtime만 사용 (inference.py에서 MODEL_TYPE='onnx')")
        
        import tempfile
        import tensorflow as tf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_path = os.path.join(tmpdir, "saved_model")
            
            if use_onnx2tf:
                print("\n[1/3] ONNX 모델 로드 중...")
                print("[2/3] TensorFlow 모델로 변환 중 (onnx2tf 사용)...")
                # onnx2tf를 사용하여 변환
                convert(
                    input_onnx_file_path=onnx_path,
                    output_folder_path=saved_model_path,
                    copy_onnx_input_output_names_to_tflite=True,
                    non_verbose=True
                )
            else:
                print("\n[1/3] ONNX 모델 로드 중...")
                import onnx
                onnx_model = onnx.load(onnx_path)
                
                print("[2/3] TensorFlow 모델로 변환 중 (onnx-tf 사용)...")
                tf_rep = prepare(onnx_model)
                tf_rep.export_graph(saved_model_path)
            
            print("[3/3] TensorFlow Lite로 변환 중...")
            
            # TensorFlow Lite 변환기 생성
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            
            # 최적화 옵션 설정 (선택사항)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 변환 실행
            tflite_model = converter.convert()
            
            # 파일 저장
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB 단위
            print(f"[OK] TensorFlow Lite 변환 완료")
            print(f"  파일 크기: {file_size:.2f} MB")
            print(f"  파일 경로: {output_path}")
            
    except ImportError as e:
        print(f"\n[ERROR] 필요한 패키지가 설치되지 않았습니다: {e}")
        print("\n필요한 패키지 설치:")
        print("  pip install onnx2tf tensorflow")
        print("\n또는 (대안):")
        print("  pip install onnx-tf tensorflow")
        raise
    except Exception as e:
        print(f"[ERROR] 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    # 설정 파라미터
    ONNX_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.onnx"
    TFLITE_OUTPUT_PATH = "./qdeep_training_output_limited/best_qdeep_model_limited.tflite"
    
    try:
        convert_onnx_to_tflite(
            onnx_path=ONNX_PATH,
            output_path=TFLITE_OUTPUT_PATH
        )
        print("\n--- [변환 완료] ---")
        print(f"ONNX 모델: {ONNX_PATH}")
        print(f"TensorFlow Lite 모델: {TFLITE_OUTPUT_PATH}")
        print("\nTensorFlow Lite 모델 사용 방법:")
        print("  import tensorflow as tf")
        print("  interpreter = tf.lite.Interpreter(model_path='모델경로.tflite')")
        print("  interpreter.allocate_tensors()")
        print("  input_details = interpreter.get_input_details()")
        print("  output_details = interpreter.get_output_details()")
        print("  interpreter.set_tensor(input_details[0]['index'], input_data)")
        print("  interpreter.invoke()")
        print("  output = interpreter.get_tensor(output_details[0]['index'])")
    except Exception as e:
        print(f"\n[치명적 오류] 변환 실패: {e}")
        exit(1)

