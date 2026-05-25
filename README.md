# capstone

## 현재 기능
- Orbbec Astra RGB/Depth 입력 기반 사람 인식
- YOLOv8n NCNN + ByteTrack 기반 target tracking
- target 거리(`depth_m`) 및 수평 방향각(`angle_x`) 계산
- Raspberry Pi SocketCAN(`can0`)으로 Teensy에 target 상태 송신

## 폴더 구조
- `service_module/person_tracking/src/person_tracking/main.py`: 추적 메인 실행
- `service_module/person_tracking/src/person_tracking/can_sender.py`: CAN 송신 모듈
- `service_module/person_tracking/config/config.yaml`: 경로/파라미터 설정
- `service_module/person_tracking/config/bytetrack.yaml`: ByteTrack 설정
- `service_module/person_tracking/models/yolov8n_ncnn_model/`: YOLO NCNN 모델

## Raspberry Pi CAN 인터페이스 준비
```bash
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
ip -details -statistics link show can0
```

테스트 종료 후:
```bash
sudo ip link set can0 down
```

## Python 실행 방법
```bash
cd service_module/person_tracking
python3 yolo_test.py
```

현재 저장소 구조에서는 패키지 실행을 사용한다:
```bash
cd service_module/person_tracking
python3 -m person_tracking.main
```

`OPENNI_PATH` 환경변수로 OpenNI 경로 override 가능:
```bash
cd service_module/person_tracking
OPENNI_PATH=/path/to/openni2/plugins python3 -m person_tracking.main
```

## CAN 프레임 규격
- CAN ID: `0x101` (standard)
- DLC: `8`
- Byte order: Little Endian
- Payload format: `struct.pack("<BBHhh", status, sequence, distance_mm, angle_cdeg, track_id)`

| Byte | 이름 | 형식 | 의미 |
|---|---|---|---|
| 0 | status | uint8 | 상태 비트 플래그 |
| 1 | sequence | uint8 | 0~255 rollover |
| 2-3 | distance_mm | uint16 LE | 대상 거리(mm) |
| 4-5 | angle_cdeg | int16 LE | 방향각×100 (0.01deg) |
| 6-7 | track_id | int16 LE | target track ID, 없으면 -1 |

status 비트:
- bit0: `TARGET_VALID`
- bit1: `DEPTH_VALID`
- bit2~7: reserved(0)

## 상태 처리 규칙
- 정상 target + 정상 depth: `status=0x03`
- target 있음 + depth invalid: `status=0x01` (`distance_mm=0`, angle/track_id는 현재 값)
- target 없음: `status=0x00` (`distance_mm=0`, `angle_cdeg=0`, `track_id=-1`)

> Teensy 측은 정상 프레임이 일정 시간 수신되지 않으면 정지하는 failsafe를 반드시 구현해야 한다 (이번 단계에서는 Teensy 제어 코드 미포함).

## CAN 송신 확인 방법
별도 터미널에서:
```bash
candump can0
```

예시(정상 target: depth=1.842m, angle=-12.35°, track_id=3, status=0x03):
- `distance_mm=1842 (0x0732)`
- `angle_cdeg=-1235 (0xFB2D, int16)`
- payload bytes: `03 seq 32 07 2D FB 03 00`

## 의존성 메모
- Python 패키지: `service_module/person_tracking/requirements.txt`
- OpenNI2/primesense는 시스템 환경마다 설치 방식이 다르므로 별도 설치 필요
