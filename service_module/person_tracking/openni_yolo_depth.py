import time
import numpy as np
import cv2

from primesense import openni2
from ultralytics import YOLO


OPENNI_PATH = "/home/dgdg/astra_sdk/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64/lib/Plugins/openni2"
MODEL_PATH = "/home/dgdg/person_tracking/yolov8n_ncnn_model"

ROI_HALF = 5          # bbox center 기준 11x11 depth ROI
MIN_DEPTH_MM = 300    # 너무 가까운 노이즈 제거
MAX_DEPTH_MM = 6000   # Astra 실사용 범위 제한


def median_depth_mm(depth_img, cx, cy):
    h, w = depth_img.shape

    x1 = max(0, cx - ROI_HALF)
    x2 = min(w, cx + ROI_HALF + 1)
    y1 = max(0, cy - ROI_HALF)
    y2 = min(h, cy + ROI_HALF + 1)

    roi = depth_img[y1:y2, x1:x2]
    valid = roi[(roi >= MIN_DEPTH_MM) & (roi <= MAX_DEPTH_MM)]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def main():
    openni2.initialize(OPENNI_PATH)

    dev = openni2.Device.open_any()
    print("Device opened:", dev.get_device_info())

    color_stream = dev.create_color_stream()
    depth_stream = dev.create_depth_stream()

    color_stream.start()
    depth_stream.start()

    model = YOLO(MODEL_PATH, task="detect")
    print("YOLO model loaded")

    frame_count = 0
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            color_frame = color_stream.read_frame()
            depth_frame = depth_stream.read_frame()

            cw, ch = color_frame.width, color_frame.height
            dw, dh = depth_frame.width, depth_frame.height

            color_data = color_frame.get_buffer_as_uint8()
            depth_data = depth_frame.get_buffer_as_uint16()

            rgb = np.frombuffer(color_data, dtype=np.uint8).reshape((ch, cw, 3)).copy()
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((dh, dw)).copy()

            results = model.predict(
                source=rgb,
                imgsz=320,
                classes=[0],
                verbose=False
            )

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            persons = len(results[0].boxes)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # color/depth 해상도가 다르면 좌표 스케일 보정
                dx = int(cx * dw / cw)
                dy = int(cy * dh / ch)

                depth_mm = median_depth_mm(depth, dx, dy)

                if depth_mm is None:
                    depth_text = "depth: invalid"
                else:
                    depth_m = depth_mm / 1000.0
                    depth_text = f"depth: {depth_m:.2f} m"

                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(bgr, (cx, cy), 4, (0, 0, 255), -1)

                cv2.putText(
                    bgr,
                    f"person {conf:.2f}",
                    (x1, max(20, y1 - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                cv2.putText(
                    bgr,
                    depth_text,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

                print(f"person conf={conf:.2f} center=({cx},{cy}) depth={depth_text}")

            frame_count += 1
            now = time.time()

            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"FPS: {fps:.1f} | persons: {persons}")

                frame_count = 0
                last_time = now

            cv2.putText(
                bgr,
                f"FPS: {fps:.1f} | persons: {persons}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Astra YOLO Depth", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        color_stream.stop()
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()