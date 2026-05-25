# person_tracking

## 목적
Raspberry Pi 5 + Orbbec Astra + YOLOv8n NCNN 기반 사람 추적 모듈입니다.

현재 선택된 target의 `depth_m`, `angle_x`를 계산하며, CAN 송신 활성화 시 해당 값을 `can0`으로 전송합니다.

## 하드웨어
- Raspberry Pi 5
- Orbbec Astra (OpenNI2)
- MCP2515 기반 CAN 모듈 (SocketCAN `can0`)

## 파일 구조
- `src/person_tracking/`: 실행 및 모듈 코드
- `src/person_tracking/can_sender.py`: 타겟 상태 CAN 송신 모듈
- `config/`: `config.yaml`, `bytetrack.yaml`
- `models/yolov8n_ncnn_model/`: YOLOv8n NCNN 모델

## 실행 환경
- conda 환경 `dgdg311`

## 설치
```bash
cd ~/capstone/service_module/person_tracking
python -m pip install -e .
pip install python-can
```

## SocketCAN 상태 확인
```bash
ip link show can0
```

## can0 활성화 예시
```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

## CAN 송신 모니터링
```bash
candump can0
```

## 실행
```bash
DISPLAY=:0 XAUTHORITY=/home/dgdg/.Xauthority python -m person_tracking.main
DISPLAY=:0 XAUTHORITY=/home/dgdg/.Xauthority python -m person_tracking.main --no-can
DISPLAY=:0 XAUTHORITY=/home/dgdg/.Xauthority python -m person_tracking.main --can-debug
```

- `--no-can`: CAN 송신 비활성화
- `--can-channel can0`: SocketCAN 채널 지정
- `--can-bitrate 500000`: CAN bitrate 지정
- `--can-send-hz 10`: 송신 주기 제한 (최대 Hz)
- `--can-debug`: 송신 payload 디버그 로그 출력

- `q`: 종료
- `r`: target reset

> Astra 하드웨어가 없는 환경에서는 전체 실행 검증이 제한됩니다.

## CAN 프로토콜

- Arbitration ID: `0x101`
- Classic CAN, 8 byte payload
- `is_extended_id=False`
- 최대 송신 주기: 10 Hz (설정 가능)
- little-endian 패킹: `struct.pack("<HhBBBB", ...)`

| Byte | 필드 | 자료형 | 설명 |
|---|---|---|---|
| 0~1 | `distance_mm` | uint16 LE | 타겟 거리(mm), invalid 시 `0xFFFF` |
| 2~3 | `angle_cdeg` | int16 LE | 방향각 × 100 |
| 4 | `track_id` | uint8 | 타겟 ID, 없으면 `0xFF` |
| 5 | `flags` | uint8 | 상태 비트필드 |
| 6 | `confidence_pct` | uint8 | confidence × 100, 없으면 0 |
| 7 | `sequence` | uint8 | 송신 시도마다 0~255 순환 증가 |

### flags 정의

| Bit | 의미 |
|---|---|
| bit 0 | `TARGET_VALID` |
| bit 1 | `DEPTH_VALID` |
| bit 2 | `ANGLE_VALID` |
| bit 3 | `TARGET_RECONNECTED` |
| bit 4~7 | 예약(0) |
