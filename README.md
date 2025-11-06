# ECG Project

## 가상 환경 설정 (pyvenv.cfg)

이 프로젝트는 Python 가상 환경을 사용합니다. `pyvenv.cfg` 파일은 가상 환경의 설정 정보를 담고 있습니다.

### pyvenv.cfg 파일 구조

```cfg
home = C:\Python313
include-system-site-packages = false
version = 3.13.3
executable = C:\Python313\python.exe
command = C:\Python313\python.exe -m venv D:\git\ECGproject
```

### 설정 항목 설명

#### `home`
- **설명**: 가상 환경을 생성한 기본 Python 설치 경로
- **예시**: `C:\Python313`
- **용도**: 가상 환경이 참조하는 시스템 Python의 위치를 지정

#### `include-system-site-packages`
- **설명**: 시스템 사이트 패키지를 가상 환경에 포함할지 여부
- **값**: `true` 또는 `false`
- **현재 설정**: `false` (시스템 패키지 제외)
- **권장**: 일반적으로 `false`로 설정하여 프로젝트 격리 유지

#### `version`
- **설명**: 가상 환경을 생성한 Python 버전
- **예시**: `3.13.3`
- **용도**: 가상 환경이 사용하는 Python 버전 정보

#### `executable`
- **설명**: 가상 환경을 생성한 Python 실행 파일의 전체 경로
- **예시**: `C:\Python313\python.exe`
- **용도**: 가상 환경이 참조하는 Python 인터프리터 위치

#### `command`
- **설명**: 가상 환경을 생성할 때 사용한 명령어
- **예시**: `C:\Python313\python.exe -m venv D:\git\ECGproject`
- **용도**: 가상 환경 생성 기록 (참고용)

### 가상 환경 생성 방법

#### Windows PowerShell
```powershell
# Python 3.13을 사용하여 가상 환경 생성
C:\Python313\python.exe -m venv .

# 또는 현재 디렉토리에 가상 환경 생성
python -m venv .
```

#### 가상 환경 활성화
```powershell
# PowerShell에서
.\Scripts\Activate.ps1

# 또는 CMD에서
.\Scripts\activate.bat
```

#### 가상 환경 비활성화
```powershell
deactivate
```

### 주의사항

1. **자동 생성 파일**: `pyvenv.cfg`는 `venv` 모듈로 가상 환경을 생성할 때 자동으로 생성됩니다.
2. **수동 수정**: 일반적으로 이 파일을 수동으로 수정할 필요는 없습니다.
3. **경로 변경**: Python 설치 경로가 변경된 경우, `home`과 `executable` 경로를 업데이트해야 할 수 있습니다.
4. **프로젝트 이식성**: 다른 시스템으로 프로젝트를 옮길 때는 해당 시스템의 Python 경로에 맞게 설정이 자동으로 조정됩니다.

### 문제 해결

#### 가상 환경이 작동하지 않는 경우
1. `home`과 `executable` 경로가 올바른지 확인
2. 해당 경로에 Python이 실제로 설치되어 있는지 확인
3. 가상 환경을 다시 생성:
   ```powershell
   Remove-Item -Recurse -Force .\Scripts, .\Lib, .\Include, .\pyvenv.cfg
   C:\Python313\python.exe -m venv .
   ```

