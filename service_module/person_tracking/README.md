# person_tracking

## 목적
Raspberry Pi 5 + Orbbec Astra + YOLOv8n NCNN 기반 사람 추적 모듈입니다.

현재 target의 `depth_m`, `angle_x`를 출력합니다.

## 하드웨어
- Raspberry Pi 5
- Orbbec Astra (OpenNI2)

## 파일 구조
- `src/person_tracking/`: 실행 및 모듈 코드
- `config/`: `config.yaml`, `bytetrack.yaml`
- `models/yolov8n_ncnn_model/`: YOLOv8n NCNN 모델

## 실행 환경
- conda 환경 `dgdg311`

## 설치
```bash
cd ~/capstone/service_module/person_tracking
python -m pip install -e .
```

## 실행 (GUI)
```bash
DISPLAY=:0 XAUTHORITY=/home/dgdg/.Xauthority python -m person_tracking.main
```

- `q`: 종료
- `r`: target reset

> Astra 하드웨어가 없는 환경에서는 실제 실행 검증이 불가능합니다.
